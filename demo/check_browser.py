"""Optional browser QA. Requires playwright and its Chromium installation."""
from pathlib import Path
from playwright.sync_api import sync_playwright

folder = Path(__file__).resolve().parent
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    for url in ("http://127.0.0.1:8765", (folder/"demo.html").as_uri()):
        page.goto(url)
        page.wait_for_function("document.getElementById('tick').textContent === '0 / 40'")
        assert page.locator("#q-values tr").count() == 5
        assert "pilot/models/dqn" in page.locator("#checkpoint").text_content()
        page.locator("#step").click()
        assert page.locator("#tick").inner_text() == "1 / 40"
        page.locator("#back").click()
        page.locator('#arm-details > summary').click()
        page.locator("#arm").select_option("2")
        page.locator('#arm-details > summary').click()
        page.locator("#candidates").uncheck()
        page.locator("#play").click()
        page.wait_for_function("document.getElementById('tick').textContent !== '0 / 40'")
        page.locator("#play").click()
        for policy in ("dqn_no_wind", "random", "greedy", "wind_heuristic", "no_intervention", "dqn"):
            page.locator("#policy").select_option(policy)
            page.wait_for_function("document.getElementById('reset').textContent === 'Load episode'")
            assert page.locator("#error").is_hidden()
            assert page.locator("#tick").inner_text() == "0 / 40"
            if policy == "dqn_no_wind":
                assert page.locator("#observed-wind").inner_text() == "0.00, 0.00"
        page.locator('#model > summary').click()
        assert "learned parameters" in page.locator("#parameters").inner_text()
        assert "QNetwork" in page.locator("#architecture").text_content()
        page.locator('#model > summary').click()
        page.locator('#results > summary').click()
        page.wait_for_function("document.querySelectorAll('#comparison tbody tr').length === 6")
        page.locator('#results > summary').click()
        page.locator("#candidates").check()
        page.screenshot(path=str(folder/"preview.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(folder/"preview-mobile.png"), full_page=True)
        page.set_viewport_size({"width": 1400, "height": 1000})
    assert not errors, errors
    browser.close()
print("Live and offline demo passed: six policies, stepping, autoplay, arm inspection, model/results, mobile layout; no JavaScript errors.")
