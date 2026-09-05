const { chromium } = require('playwright');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');
const path = require('node:path');
const outputDirectory = path.resolve(process.argv[2] || 'ci-out/workbench');

(async () => {
  const browser = await chromium.launch({ channel: process.env.GAT_BROWSER_CHANNEL || undefined, headless: true, args: ['--enable-unsafe-swiftshader'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1365, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.goto(pathToFileURL(path.join(outputDirectory, 'workbench.html')).href);
    await page.locator('#modes button[data-mode="GRAPH"]').click();
    await page.locator('g.node').filter({ hasText: 'Wall-Party' }).click();
    await page.waitForFunction(() => document.querySelector('#selection-cell').textContent.includes('Wall-Party'));
    await page.locator('#modes button[data-mode="STRUCTURE"]').click();
    const viewer = page.frameLocator('#structure');
    await viewer.locator('#selected').filter({ hasText: 'Wall-Party' }).waitFor({ state: 'visible' });
    assert.match(await viewer.locator('#meta').textContent(), /placements are exact metadata/);
    await viewer.locator('#explode').focus();
    await viewer.locator('#explode').press('End');
    assert.equal(await viewer.locator('#explodeValue').textContent(), 'exploded');
    assert.match(await viewer.locator('#selected').textContent(), /for reading/);
    assert.equal(await viewer.locator('#auditBox').isVisible(), true);
    await page.screenshot({ path: path.join(outputDirectory, 'integration-preview.png') });
    await page.locator('#modes button[data-mode="EVIDENCE"]').click();
    assert.match(await page.locator('#panels').innerText(), /REJECT/);
    await page.locator('#modes button[data-mode="STATE"]').click();
    assert.match(await page.locator('#entity-card').innerText(), /Wall-Party/i);
    await page.setViewportSize({ width: 700, height: 900 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ result: 'PASS', checks: ['graph-to-viewer selection', 'frame honesty', 'explode slider', 'audit outline availability', 'evidence verdict', 'state selection', '700px layout'], consoleErrors: errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

