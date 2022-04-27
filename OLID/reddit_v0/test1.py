from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

global driver

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://www.python.org")
print(driver.title)