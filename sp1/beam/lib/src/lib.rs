//! Shared arithmetic contract for GAT's bounded SP1 beam proof.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SP1_VERSION: &str = "6.5.0";
pub const PROOF_TYPE: &str = "core-v6.5.0";
pub const REQUEST_FORMAT: &str = "gat-sp1-beam-request-v1";
pub const RECEIPT_FORMAT: &str = "gat-sp1-beam-proof-receipt-v1";
pub const SCHEMA_VERSION: u64 = 1;
pub const NUMERIC_PROFILE_ID: &str = "beam-milli-mpa-mm3-milli-nmm-v1";
pub const NUMERIC_PROFILE_DIGEST: [u8; 32] = [
    0xf9, 0x35, 0xac, 0x37, 0xc9, 0x23, 0xc6, 0xf4, 0x7a, 0x21, 0xe9, 0xa8, 0x47, 0xd8, 0xef, 0x92,
    0xaa, 0x96, 0xf2, 0xd7, 0xe7, 0xb5, 0xc2, 0x5c, 0xe3, 0x17, 0xb4, 0x78, 0xd6, 0x03, 0x3f, 0x76,
];

const CLAIM_DOMAIN: &[u8] = b"gat-sp1-beam-claim-v1\0";
const PUBLIC_DOMAIN: &[u8] = b"gat-sp1-beam-public-v1\0";
const ONE_MILLION: u128 = 1_000_000;
const REQUIRED_PHI_PPM: u64 = 900_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeamClaimInput {
    pub yield_strength_milli_mpa: u64,
    pub plastic_section_modulus_mm3: u64,
    pub factored_demand_milli_n_mm: u64,
    pub resistance_factor_ppm: u64,
    pub numeric_profile_digest: [u8; 32],
    pub model_contract_digest: [u8; 32],
    pub validation_profile_digest: [u8; 32],
    pub evidence_digest: [u8; 32],
    pub evidence_source_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuestInput {
    pub public_statement_digest: [u8; 32],
    pub claim: BeamClaimInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeamClaimOutput {
    pub nominal_milli_n_mm: u128,
    pub available_milli_n_mm: u128,
    pub verdict: String,
    pub computation_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicValues {
    pub public_statement_digest: [u8; 32],
    pub computation_result_digest: [u8; 32],
    pub nominal_milli_n_mm: u128,
    pub available_milli_n_mm: u128,
    pub factored_demand_milli_n_mm: u128,
    pub passed: bool,
}

impl PublicValues {
    pub fn from_claim(
        statement: [u8; 32],
        input: &BeamClaimInput,
        output: &BeamClaimOutput,
    ) -> Self {
        Self {
            public_statement_digest: statement,
            computation_result_digest: output.computation_digest,
            nominal_milli_n_mm: output.nominal_milli_n_mm,
            available_milli_n_mm: output.available_milli_n_mm,
            factored_demand_milli_n_mm: input.factored_demand_milli_n_mm as u128,
            passed: output.verdict == "PASS",
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut value = Vec::with_capacity(PUBLIC_DOMAIN.len() + 32 + 32 + 16 * 3 + 1);
        value.extend_from_slice(PUBLIC_DOMAIN);
        value.extend_from_slice(&self.public_statement_digest);
        value.extend_from_slice(&self.computation_result_digest);
        value.extend_from_slice(&self.nominal_milli_n_mm.to_be_bytes());
        value.extend_from_slice(&self.available_milli_n_mm.to_be_bytes());
        value.extend_from_slice(&self.factored_demand_milli_n_mm.to_be_bytes());
        value.push(u8::from(self.passed));
        value
    }
}

pub fn evaluate(input: &BeamClaimInput) -> Result<BeamClaimOutput, &'static str> {
    if input.yield_strength_milli_mpa == 0 {
        return Err("yield strength must be positive");
    }
    if input.plastic_section_modulus_mm3 == 0 {
        return Err("plastic section modulus must be positive");
    }
    if input.resistance_factor_ppm != REQUIRED_PHI_PPM {
        return Err("the v1 guest requires phi_b = 900000 ppm");
    }
    if input.numeric_profile_digest != NUMERIC_PROFILE_DIGEST {
        return Err("numeric profile digest is not the v1 profile");
    }

    let nominal = (input.yield_strength_milli_mpa as u128)
        .checked_mul(input.plastic_section_modulus_mm3 as u128)
        .ok_or("nominal capacity overflow")?;
    let numerator = nominal
        .checked_mul(input.resistance_factor_ppm as u128)
        .ok_or("available capacity numerator overflow")?;
    let available = round_div_ties_even(numerator, ONE_MILLION)?;
    let passed = available >= input.factored_demand_milli_n_mm as u128;

    let mut hasher = Sha256::new();
    hasher.update(CLAIM_DOMAIN);
    hasher.update(input.yield_strength_milli_mpa.to_be_bytes());
    hasher.update(input.plastic_section_modulus_mm3.to_be_bytes());
    hasher.update(input.factored_demand_milli_n_mm.to_be_bytes());
    hasher.update(input.resistance_factor_ppm.to_be_bytes());
    hasher.update(input.numeric_profile_digest);
    hasher.update(input.model_contract_digest);
    hasher.update(input.validation_profile_digest);
    hasher.update(input.evidence_digest);
    hasher.update(input.evidence_source_digest);
    hasher.update(nominal.to_be_bytes());
    hasher.update(available.to_be_bytes());
    hasher.update([u8::from(passed)]);
    let computation_digest: [u8; 32] = hasher.finalize().into();

    Ok(BeamClaimOutput {
        nominal_milli_n_mm: nominal,
        available_milli_n_mm: available,
        verdict: if passed { "PASS" } else { "FAIL" }.to_string(),
        computation_digest,
    })
}

fn round_div_ties_even(numerator: u128, denominator: u128) -> Result<u128, &'static str> {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let twice = remainder.checked_mul(2).ok_or("rounding overflow")?;
    if twice > denominator || (twice == denominator && quotient % 2 == 1) {
        quotient.checked_add(1).ok_or("rounded value overflow")
    } else {
        Ok(quotient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(value: &[u8]) -> String {
        value.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn cross_language_known_vector_matches_python() {
        let input = BeamClaimInput {
            yield_strength_milli_mpa: 325_000,
            plastic_section_modulus_mm3: 1_000_000,
            factored_demand_milli_n_mm: 301_000_000_000,
            resistance_factor_ppm: 900_000,
            numeric_profile_digest: NUMERIC_PROFILE_DIGEST,
            model_contract_digest: [0x11; 32],
            validation_profile_digest: [0x22; 32],
            evidence_digest: [0x33; 32],
            evidence_source_digest: [0x44; 32],
        };
        let output = evaluate(&input).unwrap();
        assert_eq!(output.nominal_milli_n_mm, 325_000_000_000);
        assert_eq!(output.available_milli_n_mm, 292_500_000_000);
        assert_eq!(output.verdict, "FAIL");
        assert_eq!(
            hex(&output.computation_digest),
            "1443b90bc95f146a0a4c1e8e4beeb7db9c7cd59e9431f05e91958bb6c97e54e6"
        );
        let public = PublicValues::from_claim([0x55; 32], &input, &output).encode();
        assert_eq!(public.len(), 136);
        assert_eq!(
            hex(&public),
            "6761742d7370312d6265616d2d7075626c69632d76310055555555555555555555555555555555555555555555555555555555555555551443b90bc95f146a0a4c1e8e4beeb7db9c7cd59e9431f05e91958bb6c97e54e600000000000000000000004bab8272000000000000000000000000441a5bcd0000000000000000000000004614ff820000"
        );
    }

    #[test]
    fn ties_round_to_even() {
        assert_eq!(round_div_ties_even(2_500_000, ONE_MILLION).unwrap(), 2);
        assert_eq!(round_div_ties_even(3_500_000, ONE_MILLION).unwrap(), 4);
    }

    #[test]
    fn rejects_profile_drift() {
        let input = BeamClaimInput {
            yield_strength_milli_mpa: 1,
            plastic_section_modulus_mm3: 1,
            factored_demand_milli_n_mm: 0,
            resistance_factor_ppm: REQUIRED_PHI_PPM,
            numeric_profile_digest: [0; 32],
            model_contract_digest: [0; 32],
            validation_profile_digest: [0; 32],
            evidence_digest: [0; 32],
            evidence_source_digest: [0; 32],
        };
        assert_eq!(
            evaluate(&input),
            Err("numeric profile digest is not the v1 profile")
        );
    }
}
