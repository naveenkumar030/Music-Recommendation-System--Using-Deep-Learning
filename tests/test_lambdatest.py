"""
LambdaTest Automated Cloud Integration Tests.
Executes remote cloud tests on LambdaTest Selenium Grid.
"""

import time
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By

LT_USERNAME = "2411cs020069"
LT_ACCESS_KEY = "LT_9WPAT5KKdos7GpsTpC351d7gMx5ouscFbv3TNs1DlaonWpr"
HUB_URL = f"https://{LT_USERNAME}:{LT_ACCESS_KEY}@hub.lambdatest.com/wd/hub"


@pytest.fixture(scope="function")
def remote_driver():
    options = ChromeOptions()
    options.browser_version = "latest"
    options.platform_name = "Windows 11"
    lt_options = {
        "username": LT_USERNAME,
        "accessKey": LT_ACCESS_KEY,
        "build": "SoundSpace RL - Pytest Suite",
        "project": "SoundSpace DeepRL",
        "name": "Pytest Cloud Grid Test",
        "w3c": True,
        "visual": True,
        "video": True,
        "network": True,
        "console": True
    }
    options.set_capability("LT:Options", lt_options)
    driver = webdriver.Remote(command_executor=HUB_URL, options=options)
    yield driver
    try:
        driver.quit()
    except Exception:
        pass


def test_lambdatest_cloud_execution(remote_driver):
    """Test remote execution on LambdaTest Selenium Hub."""
    remote_driver.get("https://example.com")
    assert "Example" in remote_driver.title
    h1 = remote_driver.find_element(By.TAG_NAME, "h1")
    assert h1.text != ""
    remote_driver.execute_script("lambda-status=passed")
