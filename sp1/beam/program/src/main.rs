#![no_main]

use gat_sp1_beam_lib::{evaluate, GuestInput, PublicValues};

sp1_zkvm::entrypoint!(main);

pub fn main() {
    let input: GuestInput = sp1_zkvm::io::read();
    let output = evaluate(&input.claim).expect("invalid bounded beam proof input");
    let public = PublicValues::from_claim(input.public_statement_digest, &input.claim, &output);
    sp1_zkvm::io::commit_slice(&public.encode());
}
