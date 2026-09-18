/* Optional browser qualification. Requires Playwright + Chromium, no runtime dependency.
 * FIELDWORK_PLAYWRIGHT=/path/to/playwright node scripts/check_output_ux.cjs /tmp/fieldwork-output-ux
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.FIELDWORK_PLAYWRIGHT || 'playwright');

(async () => {
  const directory = path.resolve(process.argv[2]);
  const fixtures = JSON.parse(fs.readFileSync(path.join(directory, 'fixtures.json')));
  const browser = await chromium.launch({headless: true});
  const errors = [], checks = [], requests = [], bounds = [];
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (!request.url().startsWith('file:')) requests.push(request.url());
  });
  async function open(name) { await page.goto(pathToFileURL(path.join(directory, name + '.html')).href); }
  async function noOverflow(label) {
    const sizes = await page.evaluate(() => ({
      width: innerWidth, scroll: document.documentElement.scrollWidth,
      body: document.body.getBoundingClientRect().width,
    }));
    assert(sizes.scroll <= sizes.width + 1, `${label}: document overflow ${JSON.stringify(sizes)}`);
  }
  for (const name of fixtures.cases) {
    await open(name);
    const ids = await page.locator('[id]').evaluateAll(es => es.map(e => e.id));
    assert.equal(new Set(ids).size, ids.length, `${name}: duplicate IDs`);
    // Read every disclosure, including long nested evidence tables.
    await page.locator('details').evaluateAll(es => es.forEach(e => e.open = true));
    for (const width of [375, 600, 768, 1280, 1920]) {
      await page.setViewportSize({width, height: 900});
      await noOverflow(`${name}/${width}`);
      const scale = page.locator('[data-scale]');
      if (await scale.count()) {
        for (const value of ['fit', '0.5', '0.75', 'actual', '1.5', '2']) {
          await scale.selectOption(value);
          await noOverflow(`${name}/${width}/${value}`);
          const dimensions = await page.locator('.figure svg').first().evaluate(svg => ({
            width: svg.getBoundingClientRect().width,
            native: Number(svg.getAttribute('width')),
            panel: svg.parentElement.clientWidth,
          }));
          if (value === 'fit') assert(dimensions.width <= dimensions.panel + 1);
          else assert(Math.abs(dimensions.width - dimensions.native * (value === 'actual' ? 1 : Number(value))) < 2);
        }
        await scale.selectOption('actual');
      }
      const matrixView = page.locator('#view option[value="matrix"]');
      if (await matrixView.count()) {
        await page.locator('#view').selectOption('matrix');
        await noOverflow(`${name}/${width}/matrix`);
        assert(await page.locator('.grain-matrix').isVisible());
      }
      // CSS layout zoom exercises reflow at enlarged text/control sizes.
      for (const zoom of [0.75, 1.25, 2]) {
        await page.evaluate(zoom => document.documentElement.style.zoom = String(zoom), zoom);
        await noOverflow(`${name}/${width}/layout-zoom-${zoom}`);
      }
      await page.evaluate(() => document.documentElement.style.zoom = '');
      if (await matrixView.count()) {
        const matrix = page.locator('.matrix-scroll');
        await matrix.evaluate(e => { e.scrollLeft = 200; e.scrollTop = 150; });
        const sticky = await matrix.evaluate(e => ({
          container: e.getBoundingClientRect().x,
          cell: e.querySelector('tbody th').getBoundingClientRect().x,
          top: e.getBoundingClientRect().y,
          header: e.querySelector('thead th').getBoundingClientRect().y,
          scrolled: e.scrollTop,
        }));
        assert(Math.abs(sticky.cell - sticky.container) < 3, `${name}: row label not sticky`);
        if (sticky.scrolled > 60) assert(Math.abs(sticky.header - sticky.top) < 3, `${name}: column label not sticky`);
        if (name === 'wide-grain' && [375, 1280].includes(width)) {
          await matrix.scrollIntoViewIfNeeded();
          await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
          const painted = await matrix.evaluate(e => {
            const header = e.querySelector('thead th'), rect = header.getBoundingClientRect();
            return header.contains(document.elementFromPoint(rect.x + 20, rect.y + 15));
          });
          assert(painted, `${name}: sticky header is covered`);
          await page.screenshot({path: path.join(directory, `wide-matrix-${width}.png`)});
        }
        await page.locator('#matrix-search').fill('no such feature');
        assert.equal(await page.locator('[data-matrix-feature]:visible').count(), 0);
        await page.locator('#matrix-reset').click();
        await page.locator('#matrix-key').selectOption('0');
        assert.equal(await page.locator('thead [data-matrix-key]:visible').count(), 1);
        await page.locator('#matrix-reset').click();
        await page.locator('.grain-matrix tbody button').first().click();
        assert(await page.locator('#feature').inputValue());
        await page.locator('#view').selectOption('map');
      }
      checks.push({name, width});
    }
    // SVG labels must lie inside their viewBox; pan/zoom cannot recover clipped labels.
    const clipped = await page.locator('svg').evaluateAll(svgs => svgs.flatMap(svg => {
      if (!svg.getBoundingClientRect().width) return [];
      const box = svg.viewBox.baseVal;
      return [...svg.querySelectorAll('text')].filter(text => {
        const b = text.getBBox();
        return b.x < -1 || b.y < -1 || b.x + b.width > box.width + 1 || b.y + b.height > box.height + 1;
      }).map(text => text.textContent);
    }));
    if (clipped.length) bounds.push({name, clipped});
    if (['wide-missingness', 'laboratory', 'long-discovery', 'long-joint', 'deep-census'].includes(name)) {
      await page.setViewportSize({width: 375, height: 900});
      await page.screenshot({path: path.join(directory, name + '-mobile.png')});
      await page.setViewportSize({width: 1280, height: 900});
      await page.screenshot({path: path.join(directory, name + '-desktop.png')});
    }
  }
  fs.writeFileSync(path.join(directory, 'svg-bounds.json'), JSON.stringify(bounds, null, 2));
  await open('overview');
  const findings = page.locator('[data-collection]').filter({has: page.locator('[data-pattern-filter]')});
  await findings.locator('[data-search]').fill('no possible match');
  assert.equal(await findings.locator('[data-record]:visible').count(), 0);
  await findings.locator('[data-reset]').click();
  const patterns = await findings.locator('[data-pattern-filter] option').evaluateAll(es => es.map(e => e.value));
  await findings.locator('[data-pattern-filter]').selectOption(patterns[1]);
  assert(await findings.locator('[data-record]:visible').count() > 0);
  await findings.locator('[data-exceptions-filter]').check();
  assert(await findings.locator('[data-record]:visible').evaluateAll(es => es.every(e => e.dataset.exceptions === 'true')));
  await findings.locator('[data-reset]').click();
  await findings.locator('[data-expand]').click();
  assert(await findings.locator('[data-record]').evaluateAll(es => es.every(e => e.open)));
  await findings.locator('[data-close]').click();
  assert(await findings.locator('[data-record]').evaluateAll(es => es.every(e => !e.open)));
  const link = page.locator('a').filter({hasText: 'inspect evidence'}).first();
  const href = await link.getAttribute('href');
  await findings.locator('[data-search]').fill('no possible match');
  await page.locator('details').filter({has: link}).evaluateAll(es => es.forEach(e => e.open = true));
  await link.click();
  const target = page.locator(href);
  await target.waitFor({state: 'visible'});
  assert(await target.evaluate(e => e.open && document.activeElement === e.querySelector('summary')));
  await page.reload();
  assert(await target.evaluate(e => e.open));
  await open('laboratory');
  const feature = page.locator('[data-feature]:visible').first();
  await feature.focus(); await page.keyboard.press('Enter');
  assert.equal(await feature.getAttribute('aria-pressed'), 'true');
  await page.locator('#selection-link').click();
  assert(await page.locator('[data-evidence][open]').count() > 0);
  await page.locator('#view').selectOption('matrix');
  assert(await page.locator('#collapse').isDisabled());
  await page.locator('#reset-view').click();
  assert.equal(await page.locator('#view').inputValue(), 'map');
  assert.equal(await page.locator('#feature').inputValue(), '');
  await open('deep-census');
  const branch = page.locator('[data-collapse-row]').first();
  await branch.click(); assert.equal(await branch.getAttribute('aria-expanded'), 'false');
  assert(await page.locator('[data-tree-row][hidden]').count() > 0);
  await page.locator('#reset-view').click();
  assert.equal(await page.locator('[data-tree-row][hidden]').count(), 0);
  await open('pairs');
  await page.locator('#context').selectOption('1');
  await page.locator('#view').selectOption('association');
  assert.equal(await page.locator('[data-context]:visible').count(), 2);
  // Native details remain usable without JavaScript; unavailable controls stay hidden.
  const staticContext = await browser.newContext({javaScriptEnabled: false, viewport: {width: 375, height: 900}});
  const staticPage = await staticContext.newPage();
  await staticPage.goto(pathToFileURL(path.join(directory, 'overview.html')).href);
  assert.equal(await staticPage.locator('[data-enhance]:visible').count(), 0);
  const staticCard = staticPage.locator('[data-pattern]').first();
  await staticCard.locator('summary').click();
  assert.equal(await staticCard.getAttribute('open'), '');
  const result = {browser: await browser.version(), fixtures, viewportChecks: checks.length,
    widths: [375, 600, 768, 1280, 1920], figureScales: ['fit', '50%', '75%', '100%', '150%', '200%'],
    layoutZoom: [0.75, 1.25, 2], errors, requests, clippedLabels: bounds};
  fs.writeFileSync(path.join(directory, 'browser-results.json'), JSON.stringify(result, null, 2) + '\n');
  await browser.close();
  assert.deepEqual(errors, [], 'JavaScript errors');
  assert.deepEqual(requests, [], 'Unexpected external requests');
  assert.deepEqual(bounds, [], 'Clipped SVG labels');
  console.log(JSON.stringify(result, null, 2));
})().catch(error => { console.error(error); process.exit(1); });
