"""Check actual recorded Firehose playback in the running local server."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8766')
args = parser.parse_args()
folder = Path(__file__).resolve().parent
with sync_playwright() as browser_tool:
    browser = browser_tool.chromium.launch()
    page = browser.new_page(viewport={'width': 1280, 'height': 960})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(args.url)
    page.wait_for_function("!document.getElementById('viewer').hidden || !document.getElementById('message').textContent.startsWith('Loading')")
    assert page.locator('#viewer').is_visible(), page.locator('#message').inner_text()
    assert page.locator('#checkpoint option').count() > 0
    assert page.locator('#seed option').count() > 0
    assert page.locator('#results tbody tr').count() > 0
    initial = page.locator('#interval').inner_text()
    if page.locator('#next').is_enabled():
        page.locator('#next').click()
        assert page.locator('#interval').inner_text() != initial
        page.locator('#back').click()
        assert page.locator('#interval').inner_text() == initial
        page.locator('#speed').select_option('150')
        page.locator('#play').click()
        page.wait_for_function("Number(document.getElementById('timeline').value) > 0")
        if page.locator('#play').inner_text() == 'Pause':
            page.locator('#play').click()
    for checkpoint in page.locator('#checkpoint option').evaluate_all('(nodes) => nodes.map(n => n.value)'):
        page.locator('#checkpoint').select_option(checkpoint)
        assert page.locator('#interval').inner_text().startswith('0 /')
        assert page.locator('#left-subtitle').inner_text()
        for seed in page.locator('#seed option').evaluate_all('(nodes) => nodes.map(n => n.value)'):
            page.locator('#seed').select_option(seed)
            for baseline in page.locator('#baseline option').evaluate_all('(nodes) => nodes.map(n => n.value)'):
                page.locator('#baseline').select_option(baseline)
                assert page.locator('#interval').inner_text().startswith('0 /')
    wait_options = page.locator('#baseline option').evaluate_all('(nodes) => nodes.filter(n => /no.intervention/i.test(n.textContent)).map(n => n.value)')
    if wait_options:
        page.locator('#baseline').select_option(wait_options[0])
        if page.locator('#next').is_enabled():
            page.locator('#next').click()
            assert 'Wait / none' in page.locator('#right-action').inner_text()
            assert 'cell ID -1' not in page.locator('#right-action').inner_text()
    page.locator('#trajectory-details > summary').click()
    assert page.locator('#trajectory tbody tr').count() > 0
    page.locator('#trajectory-details > summary').click()
    page.locator('#source-details > summary').click()
    assert page.locator('#checkpoint-detail').inner_text()
    page.locator('#source-details > summary').click()
    page.screenshot(path=str(folder/'preview.png'), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(folder/'preview-mobile.png'), full_page=True)
    assert not errors, errors
    browser.close()
print('Firehose playback passed: recorded checkpoint/seed/baseline choices, stepping, autoplay, trajectory, provenance, mobile layout, no JavaScript errors.')
