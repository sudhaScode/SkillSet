
# Selenium Automation
- `Web Testing:` Selenium can automate browser interactions to perform repetitive testing tasks, ensuring web applications function correctly across different browsers.
- `Web Scraping:` Selenium can be used to extract data from websites by simulating user actions and collecting the resulting content.

**Core Components**
- `Selenium WebDriver:` This is the primary component that allows you to control a web browser through code. It provides a language-neutral API that can be used with various programming languages.
- `Selenium IDE:` This is a browser extension for Chrome, Firefox, and Edge that simplifies recording and playback of browser interactions. It's a good tool for beginners to learn the basics of Selenium.
- `Selenium Grid:` This component allows you to distribute tests across multiple machines and run them on different browsers and operating systems.

# Get started
Steps to Automating the web application for fucntional testing
## Choosing a Browser and Language Bindings:
Selenium supports various web browsers through browser-specific drivers. Here's how to choose the right one:

- `Target Browser:` Identify which browser(s) your application primarily targets (Chrome, Firefox, Edge, etc.).
- `Language Bindings:` Selenium provides language bindings for Python. You'll need the specific driver for your chosen browser that works with Python.

**Selwcting web drivers**<br>

`Chrome:` The most popular choice. You'll need the ChromeDriver:
```
from selenium import webdriver

driver = webdriver.Chrome()  # Assuming you have ChromeDriver downloaded and in PATH

```
`Firefox:` Another popular option. You'll need the GeckoDriver:

```
from selenium import webdriver

driver = webdriver.Firefox()  # Assuming you have GeckoDriver downloaded and in PATH

```
**Configure Web driver options**<br>
WebDriver options allow you to configure the browser's behavior during tests. Here are some common options:

- `Headless Mode (--headless):` Runs the browser in the background without a graphical interface. Useful for faster tests or server-side headless browser testing.
- `Disabling Extensions (--disable-extensions):` Disables browser extensions that might interfere with your tests.
- `Downloading Specific Preferences:` Sets browser preferences from a JSON file.

```
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Create Chrome options object
options = Options()

# Add desired options (can add more options here)
options.add_argument("--headless")  # Run headless
options.add_argument("--disable-extensions")  # Disable extensions
options.add_experimental_option("prefs", {"download.default_directory": "/path/to/downloads"})  # Set download directory

# Initialize Chrome driver with options
driver = webdriver.Chrome(options=options)

```

```
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By // to selecting a web element

# Create Firefox options object
options = Options()

# Add desired options
options.add_argument("--headless")  # Run headless
options.add_argument("--disable-extensions")  # Disable extensions
options.add_argument("--profile=/path/to/your/firefox/profile")  # Load a specific profile
options.log_level = "debug"  # Set log level (options: trace, debug, info, etc.)

# Initialize Firefox driver with options
driver = webdriver.Firefox(options=options)


```
## Language Bindings
In Selenium, language bindings refer to libraries or packages that allow you to interact with the Selenium WebDriver API from a specific programming language. don't need to explicitly "set" language bindings within Selenium itself. Installing the appropriate library for your chosen language takes care of that.
1. Selenium WebDriver API:
Selenium WebDriver is the core component that allows you to control a web browser through code.
It provides a language-neutral API with methods for various browser interactions like navigating to URLs, finding elements, clicking buttons, and submitting forms.

2. Language Bindings:
Each supported programming language in Selenium has its own language binding library.
These libraries translate the WebDriver API calls into functions and objects that are idiomatic to the specific language.
This makes it easier and more natural to write Selenium tests in your preferred language.

3. Common Language Bindings:
Python: The selenium library is the official Python binding for Selenium WebDriver. `pip install selenium`
Java: The selenium-java library is the official Java binding. `npm install selenium-java`
JavaScript (Node.js): The @sitespro/webdriverio or selenium-webdriver libraries are popular bindings for Node.js. `npm install selenium-webdriver`
C#: The Selenium.WebDriver library provides C# bindings for Selenium.
Many more: Selenium supports bindings for various other languages like Ruby, PHP, etc.

<br>

Benefits of Language Bindings:
 - Readability and Maintainability: Test scripts written in your preferred language are easier to understand and maintain compared to using the raw WebDriver API.
 - Improved Developer Experience: Language bindings often provide additional features and conveniences specific to the language, making development more efficient.
 - Large Community Support: Popular language bindings like the Python library have a vast community and extensive documentation, making it easier to find help and resources.
<br>

How to Set Language Bindings:<br>

Here's the general approach (specific steps may vary slightly depending on your language and environment):

- Choose your language: Decide on the programming language you'll use for your Selenium tests.
- Install the language binding library: Use your package manager (e.g., pip for Python, npm for Node.js) to install the corresponding language binding library.
- Import the library in your code: Import the library module in your test script to start using the Selenium functionalities.


## Waiting Stratagies
Perhaps the most common challenge for browser automation is ensuring that the ``web application is in a state to execute a particular Selenium command as desired`. The processes often end up in a `race condition` where sometimes the `browser gets into the right state first` (things work as intended) and sometimes the `Selenium code executes first` (things do not work as intended). This is one of the primary causes of `flaky tests`.

**Selenium provides two different mechanisms for synchronization that are better:**

*`Implicit waits`*<br>
Selenium has a built-in way to automatically wait for elements called an implicit wait. An implicit wait value can be set either with the timeouts capability in the browser options, or with a driver method (as shown below).
<br>
This is a global setting that applies to every element location call for the entire session. The default value is 0, which means that if the element is not found, it will immediately return an error. If an implicit wait is set, the driver will wait for the duration of the provided value before returning the error. 
```
    driver.implicitly_wait(2)
```

*`Explicit waits`*<br>
Explicit waits are loops added to the code that `poll the application for a specific condition to evaluate as true before it exits` the loop and continues to the next command in the code. If the condition is not met before a designated timeout value, the code will give a timeout error. Since there are many ways for `the application not to be in the desired state`, explicit waits are a great choice to `specify the exact condition to wait for in each place it is needed`. Another nice feature is that, `by default, the Selenium Wait class automatically waits for the designated element to exist`.
```
    wait = WebDriverWait(driver, timeout=2)
    wait.until(lambda d : revealed.is_displayed())

```
*`Customization`*<br>
The Wait class can be instantiated with various parameters that will change how the conditions are evaluated.<br>
This can include:

- Changing how often the code is evaluated (polling interval)
- Specifying which exceptions should be handled automatically
- Changing the total timeout length
- Customizing the timeout message
For instance, if the element not interactable error is retried by default, then we can add an action on a method inside the code getting executed (we just need to make sure that the code returns true when it is successful):

```
    errors = [NoSuchElementException, ElementNotInteractableException]
    wait = WebDriverWait(driver, timeout=2, poll_frequency=.2, ignored_exceptions=errors)
    wait.until(lambda d : revealed.send_keys("Displayed") or True)

```
Other Expected Conditions: Selenium provides a variety of built-in expected conditions (EC) such as `visibility_of_element_located`, `element_to_be_clickable`, etc., which can be used in place of custom conditions depending on your specific needs.

## Web elements
Identifying and working with element objects in the DOM to perform events like click , send keys, double click, hover.<br>
The majority of automating web application with Selenium code involves working with web elements.<br>
1. **File Upload**<br>
Selenium cannot interact with with file upload dialog which is not part of driver, it provides the ways to upload file with out opening the dialog. <br>
If the element is an `input element with type file`, you can use the `send keys method to send the full path` to the file that will be uploaded.
```
    file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    file_input.send_keys(upload_file)
    driver.find_element(By.ID, "file-submit").click()

```
<br>

2. **Locator strategies** <br>
Ways to identify one or more specific elements in the DOM.
A locator is a way to identify elements on a page. It is the argument passed to the [https://www.selenium.dev/documentation/webdriver/elements/finders/](`Finding element`) methods.<br>

**Traditional Locators**
Selenium provides support for these 8 traditional location strategies in WebDriver:
| Locator           | Description                                                                                                                                 |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| class name        | Locates elements whose class name contains the search value (compound class names are not permitted)                                       |
| css selector      | Locates elements matching a CSS selector                                                                                                     |
| id                | Locates elements whose ID attribute matches the search value                                                                                  |
| name              | Locates elements whose NAME attribute matches the search value                                                                                |
| link text         | Locates anchor elements whose visible text matches the search value                                                                           |
| partial link text | Locates anchor elements whose visible text contains the search value. If multiple elements match, only the first one will be selected         |
| tag name          | Locates elements whose tag name matches the search value                                                                                      |
| xpath             | Locates elements matching an XPath expression                                                                                                |

<br>

1. `By ID: find_element_by_id(id)`

- Locates an element using the value of the id attribute.
```
Example: driver.find_element_by_id("element_id")

```
2. `By Class Name: find_element_by_class_name(class_name)`
Locates elements by their CSS class name.
```
Example: driver.find_element_by_class_name("some_class")

```
3. `By Name: find_element_by_name(name)`
Locates elements by the name attribute value.
```
Example: driver.find_element_by_name("element_name")

```
4. `By Tag Name: find_element_by_tag_name(tag_name)`
Locates elements by their HTML tag name.
```
Example: driver.find_element_by_tag_name("div")

```

5. `By Link Text: find_element_by_link_text(link_text)`
```
Locates anchor elements (<a>) by their exact visible text.

Example: driver.find_element_by_link_text("Click Here")
```

6. `By Partial Link Text: find_element_by_partial_link_text(link_text)`
```
Locates anchor elements (<a>) by partial visible text.

Example: driver.find_element_by_partial_link_text("Click")
```
7. `By CSS Selector: find_element_by_css_selector(css_selector)`

Locates elements using a CSS selector.
```
Example: driver.find_element_by_css_selector(".some_class")
```
8. `By XPath: find_element_by_xpath(xpath)`

Locates elements using an XPath expression.
```
Example: driver.find_element_by_xpath("//div[@class='container']")
```

**XPath is popular for several reasons when it comes to web scraping or automated testing:**
The XPath could be absolute xpath, which is created from the root of the document. Example - /html/form/input[1] Or the xpath could be relative. Example- //input[@name=‘fname’]. 

- `Precise Targeting:` XPath allows you to navigate through the HTML structure of a webpage using path expressions, which can be very specific. This precision helps in directly targeting the elements you need without relying solely on class names, IDs, or other attributes that might not be unique or stable.

- `Flexibility:` Unlike other locators that are limited to specific attributes or properties of an element (like class name or ID), XPath can traverse both upwards and downwards in the HTML tree. This flexibility allows you to select elements based on their relationship with other elements or based on more complex conditions.

- `Handling Dynamic Content:` When elements on a webpage have dynamic attributes or structures that might change, XPath can be more resilient. For example, if elements have changing IDs or classes, XPath can often still locate them based on their relative position or other attributes.

- `Powerful Functions and Predicates:` XPath supports a variety of functions and predicates that can filter and select elements based on specific criteria. This makes it possible to create very precise queries to select exactly the elements you need, even in complex scenarios.

- `Cross-browser Compatibility:` XPath is supported across different browsers, making it a reliable choice for automated testing and web scraping tasks that need to work consistently across different environments.

```
Here are some examples of selecting web elements using XPath in Python with Selenium.

# Selecting an element by ID
element = driver.find_element_by_xpath("//*[@id='element_id']")

# Selecting elements by class name
elements = driver.find_elements_by_xpath("//*[contains(@class, 'some_class')]")
for element in elements:
    print("Element by Class Name:", element.text)

# Selecting an element by attribute value
element = driver.find_element_by_xpath("//*[@name='element_name']")
print("Element by Attribute Value:", element.text)


# Selecting an element by tag name and index
element = driver.find_element_by_xpath("(//div[@class='container'])[1]")
print("Element by Tag Name and Index:", element.text)

# Selecting an anchor element by text content
element = driver.find_element_by_xpath("//a[contains(text(), 'Click Here')]")
print("Anchor Element by Text Content:", element.get_attribute('href'))


#And uisng By class need a explicit importing from `selemium.webdriver.commom.by`
# Finding an element using By.XPATH
element = driver.find_element(By.XPATH, "//a[contains(text(), 'Click Here')]")
print("Element by By.XPATH:", element.get_attribute('href'))

```
## Relative Locators
Relative locators are a feature introduced in Selenium 4, which provide a convenient way to locate elements relative to other elements on the webpage. These can be used when the specific position or relationship of an element relative to another is important for the test or automation script.
<br>

In Selenium 4, the relative locators are accessed through the selenium.webdriver.common.by.RelativeLocator class. They offer several methods to locate elements based on their proximity or relationship to a reference element. These methods include:

- `above(element)`
- `below(element)`
- `to_left_of(element)`
- `to_right_of(element)`
- `near(element)` <br>
Here's a brief explanation of each method:

- `above(element):` Finds an element that is located directly above the reference element.

- `below(element):` Finds an element that is located directly below the reference element.

- `to_left_of(element):` Finds an element that is located directly to the left of the reference element.

- `to_right_of(element):` Finds an element that is located directly to the right of the reference element.

- `near(element):` Finds an element that is located near (in any direction) the reference element.

**Example Code**
```
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.by import RelativeLocator

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the webpage
driver.get("https://example.com")

# Assume we have a reference element
reference_element = driver.find_element(By.ID, "reference_element_id")

# Using relative locators
element_above = driver.find_element(RelativeLocator.withTagName("div").above(reference_element))
element_below = driver.find_element(RelativeLocator.withTagName("div").below(reference_element))
element_left = driver.find_element(RelativeLocator.withTagName("div").toLeftOf(reference_element))
element_right = driver.find_element(RelativeLocator.withTagName("div").toRightOf(reference_element))
element_near = driver.find_element(RelativeLocator.withTagName("div").near(reference_element))

# Perform actions on located elements
print("Element Above:", element_above.text)
print("Element Below:", element_below.text)
print("Element Left:", element_left.text)
print("Element Right:", element_right.text)
print("Element Near:", element_near.text)

# Close the browser
driver.quit()

```

**Benefits of Relative Locators:**

- `Simplicity:` Relative locators provide a more intuitive way to locate elements based on their relative position to other elements, reducing the need for complex XPath or CSS selector expressions.

- `Readability:` The code becomes more readable and maintainable as it explicitly states the relationship between elements.

- `Robustness:` Relative locators can help make your tests more robust to changes in the UI layout, as they focus on the spatial relationships rather than static attributes like IDs or classes which may change.
<br>
Relative locators are particularly useful in scenarios where you need to verify the position of elements in relation to each other, such as verifying the alignment of form elements or checking the placement of UI components. They enhance the capabilities of Selenium for more sophisticated and reliable web automation scripts.

# Finding web elements -Finders
Locating the elements based on the provided locator values. One of the most fundamental aspects of using Selenium is obtaining element references to work with.
<br>

## **First matching element**
Many locators will match multiple elements on the page. The singular find element method will return a reference to the first element found within a given context.
<br>

`Evaluating entire DOM:`
When the find element method is called on the driver instance, it returns a reference to the first element in the DOM that matches with the provided locator. 
```
<ol id="vegetables">
 <li class="potatoes">…
 <li class="onions">…
 <li class="tomatoes"><span>Tomato is a Vegetable</span>…
</ol>

vegetable = driver.find_element(By.CLASS_NAME, "tomatoes")
  
```
<br>

## All matching elements
There are several use cases for needing to get references to all elements that match a locator, rather than just the first one. The plural find elements methods return a collection of element references. If there are no matches, an empty list is returned. 

```
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Firefox()

    # Navigate to Url
driver.get("https://www.example.com")

    # Get all the elements available with tag name 'p'
elements = driver.find_elements(By.TAG_NAME, 'p')

for e in elements:
    print(e.text)
  
  
```
## Find Elements From Element
It is used to find the list of matching child WebElements within the context of parent element. To achieve this, the parent WebElement is chained with ‘findElements’ to access child elements

```
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.example.com")

    # Get element with tag name 'div'
element = driver.find_element(By.TAG_NAME, 'div')

    # Get all the elements available with tag name 'p'
elements = element.find_elements(By.TAG_NAME, 'p')
for e in elements:
    print(e.text)
  
```
## Get Active Element
It is used to track (or) find DOM element which has the focus in the current browsing context.
```
  from selenium import webdriver
  from selenium.webdriver.common.by import By

  driver = webdriver.Chrome()
  driver.get("https://www.google.com")
  driver.find_element(By.CSS_SELECTOR, '[name="q"]').send_keys("webElement")

    # Get attribute of current active element
  attr = driver.switch_to.active_element.get_attribute("title")
  print(attr)
  

```

# Interacting with web elements
A high-level instruction set for manipulating form controls.
<br>

There are only 5 basic commands that can be executed on an element:
- [https://w3c.github.io/webdriver/#element-click](click) (applies to any element)
   - The element click command is executed on the center of the element. If the center of the element is obscured for some reason, Selenium will return an element click intercepted error.
   ```
   # Click on the element 
	driver.find_element(By.NAME, "color_input").click()
   ```
- [https://w3c.github.io/webdriver/#element-send-keys](send keys) (only applies to text fields and content editable elements)
   - The element send keys command types the provided keys into an editable element. Typically, this means an element is an input element of a form with a text type or an element with a content-editable attribute. If it is not editable, an invalid element state error is returned.

   ```

    # Clear field to empty it from any previous data
	driver.find_element(By.NAME, "email_input").clear()

	# Enter Text
	driver.find_element(By.NAME, "email_input").send_keys("admin@localhost.dev" )
   ```
- [https://w3c.github.io/webdriver/#element-send-keys](clear) (only applies to text fields and content editable elements)
   - The element clear command resets the content of an element. This requires an element to be editable, and resettable. Typically, this means an element is an input element of a form with a text type or an element with acontent-editable attribute. If these conditions are not met, an invalid element state error is returned.
   ```
    # Clear field to empty it from any previous data
	driver.find_element(By.NAME, "email_input").clear()
   ```
- submit (only applies to form elements)
  -  click the applicable form submission button instead.
- [https://www.selenium.dev/documentation/webdriver/support_features/select_lists/](select) (see Select List Elements)


# Information about web elements

There are a number of details you can query about a specific element.

## Is Displayed
This method is used to check if the connected Element is displayed on a webpage. Returns a Boolean value, True if the connected element is displayed in the current browsing context else returns false.
```
# Navigate to the url
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

# Get boolean value for is element display
is_email_visible = driver.find_element(By.NAME, "email_input").is_displayed()

```

## Is Enabled
This method is used to check if the connected Element is enabled or disabled on a webpage. Returns a boolean value, True if the connected element is enabled in the current browsing context else returns false.

```
    # Navigate to url
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

    # Returns true if element is enabled else returns false
value = driver.find_element(By.NAME, 'button_input').is_enabled()
  
```

## Is Selected
This method determines if the referenced Element is Selected or not. This method is widely used on Check boxes, radio buttons, input elements, and option elements.
<br>
Returns a boolean value, True if referenced element is selected in the current browsing context else returns false.

 ```
     # Navigate to url
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

    # Returns true if element is checked else returns false
value = driver.find_element(By.NAME, "checkbox_input").is_selected()
  
 ```

 ## Tag Name
 It is used to fetch the TagName of the referenced Element which has the focus in the current browsing context.

 ```
     # Navigate to url
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

    # Returns TagName of the element
attr = driver.find_element(By.NAME, "email_input").tag_name
  
 ```

##  Size and Position
It is used to fetch the dimensions and coordinates of the referenced element.<br>

The fetched data body contain the following details:<br>
- X-axis position from the top-left corner of the element
- y-axis position from the top-left corner of the element
- Height of the element
- Width of the element

```
    # Navigate to url
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

    # Returns height, width, x and y coordinates referenced element
res = driver.find_element(By.NAME, "range_input").rect
  
```
## Get CSS Value
Retrieves the value of specified computed style property of an element in the current browsing context.

```
    # Navigate to Url
driver.get('https://www.selenium.dev/selenium/web/colorPage.html')

    # Retrieves the computed style property 'color' of linktext
cssValue = driver.find_element(By.ID, "namedColor").value_of_css_property('background-color')

  
```
## Text Content
Retrieves the rendered text of the specified element.

```
    # Navigate to url
driver.get("https://www.selenium.dev/selenium/web/linked_image.html")

    # Retrieves the text of the element
text = driver.find_element(By.ID, "justanotherlink").text
  
```

## Fetching Attributes or Properties
Fetches the run time value associated with a DOM attribute. It returns the data associated with the DOM attribute or property of the element.

```
# Navigate to the url
driver.get("https://www.selenium.dev/selenium/web/inputs.html")

# Identify the email text box
email_txt = driver.find_element(By.NAME, "email_input")

# Fetch the value property associated with the textbox
value_info = email_txt.get_attribute("value")
  
```
w












# Handling Specific Use cases

## Retrieve the URL 
To retrieve the URL of the current window in Selenium WebDriver, regardless of whether the automation is interacting with a modal dialog or the main window, you can use the `current_url attribute of the WebDriver instance`. Here’s how you can do it:
```
from selenium import webdriver
# Initialize the WebDriver
driver = webdriver.Chrome()

try:
    # Open a webpage
    driver.get("https://example.com")

    # Perform actions, interact with elements, including modals or dialogs

    # Get the current URL of the window
    current_url = driver.current_url
    print("Current URL:", current_url)

finally:
    # Close the browser
    driver.quit()

```
## Handling multiple tabs (or windows)

1. Open a New Tab or Window:

When a new tab or window opens as a result of user interaction or a test step, Selenium will still manage these tabs within the same WebDriver instance.

2. Switching to the New Tab:

Use driver.switch_to.window(window_handle) to switch focus to the new tab or window. Each tab or window has a unique window_handle identifier.

3. Perform Actions in the New Tab:

Once switched to the new tab, you can continue interacting with elements and performing actions as usual.

4. Switch Back to Original Tab (if needed):

After completing actions in the new tab, you can switch back to the original tab using driver.switch_to.window(original_window_handle).
Here’s an example to illustrate switching between tabs:

```
from selenium import webdriver
import time

# Initialize the WebDriver
driver = webdriver.Chrome()

try:
    # Open the first webpage
    driver.get("https://www.google.com")
    
    # Get the current window handle (main window)
    main_window_handle = driver.current_window_handle
    
    # Open a new tab (for demonstration purposes)
    driver.execute_script("window.open('https://www.example.com', 'new_tab')")
    
    # Switch to the new tab
    new_window_handle = None
    for handle in driver.window_handles: # driver.window_handles returns a list of all open window handles (IDs).
        if handle != main_window_handle:
            new_window_handle = handle
            break
    
    if new_window_handle:
        driver.switch_to.window(new_window_handle)
        
        # Perform actions in the new tab
        print("Switched to new tab. Current URL:", driver.current_url)
        
        # Example: Get title of the new tab
        print("Title of new tab:", driver.title)
        
        # Example: Click on an element in the new tab
        # driver.find_element_by_xpath("//button[contains(text(), 'Button')]").click()
        
        # Switch back to the main window
        driver.switch_to.window(main_window_handle)
        print("Switched back to main window. Current URL:", driver.current_url)
    
finally:
    # Close the browser
    driver.quit()

```
## Handling modals in Selenium
Handling modals in Selenium involves several steps, especially when triggering them via click events or other user interactions on the webpage. Here’s a detailed example of how you can automate interactions with modals that appear after a click event using Selenium WebDriver in Python.

### Example Scenario:
Let's assume we have a webpage where clicking a button triggers a modal dialog. We'll automate the process of clicking the button, waiting for the modal to appear, interacting with elements inside the modal, and then closing the modal.


```
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the WebDriver
driver = webdriver.Chrome()

try:
    # Open the webpage
    driver.get("https://www.example.com")

    # Find and click the button that triggers the modal
    trigger_button = driver.find_element(By.ID, "open-modal-button")
    trigger_button.click()

    # Wait for the modal to appear using WebDriverWait
    wait = WebDriverWait(driver, 10)
    modal = wait.until(EC.visibility_of_element_located((By.ID, "modal-dialog")))

    # Example interaction with elements inside the modal
    modal_title = modal.find_element(By.XPATH, "//h2[contains(@class, 'modal-title')]")
    print("Modal Title:", modal_title.text)

    # Example: Close the modal (assuming there's a close button)V
    close_button = modal.find_element(By.XPATH, "//button[contains(text(), 'Close')]")
    close_button.click()

    # Alternatively, you can use JavaScript to close the modal if needed
    # driver.execute_script("arguments[0].click();", close_button)

finally:
    # Close the browser
    driver.quit()

```
### Explanation:
- `Initialize WebDriver:` Start by initializing the WebDriver instance (in this case, Chrome).

- `Navigate to Webpage:` Open the webpage where the modal can be triggered by clicking a button (https://www.example.com in this example).

- `Find and Click Trigger Button:` Locate the button that triggers the modal using find_element(By.ID, "open-modal-button") and then simulate a click with trigger_button.click().

- `Wait for Modal to Appear:` Use WebDriverWait to wait until the modal dialog becomes visible (EC.visibility_of_element_located((By.ID, "modal-dialog"))). This ensures that Selenium waits for the modal to fully load before proceeding with interactions.

- `Interact with Modal Elements:` Once the modal is visible, interact with its elements. In this example, we find the modal title using modal.find_element(By.XPATH, "//h2[contains(@class, 'modal-title')]").

- `Close the Modal:` Locate the close button within the modal (modal.find_element(By.XPATH, "//button[contains(text(), 'Close')]")) and simulate a click (close_button.click()). Alternatively, you can use JavaScript execution (driver.execute_script) to interact with elements if necessary.

- `Clean Up:` Finally, ensure to close the WebDriver instance (driver.quit()) to release resources.

## Handling iframes or Shadow DOM
Handling iframes and Shadow DOM elements in Selenium involves switching the WebDriver context to interact with elements inside these nested structures. This is crucial for testing scenarios where modals, dialogs, or other content are embedded within iframes or Shadow DOMs on a webpage.

### Automation Test Case Example:
Let's consider a scenario where you have a webpage that includes an iframe containing a modal dialog. Here’s how you can automate interactions with elements inside the iframe using Selenium:

1. initialize web driver
Start by initializing the WebDriver instance, in this case, Chrome:

```

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Initialize the WebDriver
driver = webdriver.Chrome()

```
2. navigate to web page

Open the webpage that contains the iframe with the modal dialog
```

driver.get("https://example.com")

```

3. Switch to the iframe:
Identify the iframe element using its locator (e.g., id, name, index, or by finding the element directly):
```
# Switch to the iframe by index (0-based index)
iframe = driver.find_element(By.TAG_NAME, "iframe")
driver.switch_to.frame(iframe)

# Switch to the iframe by ID or name
# driver.switch_to.frame("iframe_id_or_name")

```

4. Interact with Elements Inside the iframe:

Once inside the iframe context, you can interact with elements as if they were part of the main page:
```
# Example: Click on a button inside the iframe
iframe_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Close')]")
iframe_button.click()

```
5. Switch Back to the Main Content:
- After interacting with elements inside the iframe, switch back to the main content:
```
driver.switch_to.default_content()
driver.quit() # to close web page
```

### Handling Shadow DOM Elements:
If the modal or elements are within a Shadow DOM, the approach is somewhat different. Selenium currently does not have built-in support for interacting directly with Shadow DOM elements. You typically need to execute JavaScript to access and manipulate Shadow DOM elements. Here’s a basic outline of how you might approach it:  

```
# Example: Switching to Shadow DOM context (hypothetical approach)
shadow_host = driver.find_element(By.CSS_SELECTOR, "div.shadow-host")
shadow_root = driver.execute_script("return arguments[0].shadowRoot", shadow_host)

# Example: Finding and interacting with elements within the Shadow DOM
shadow_element = shadow_root.find_element(By.CSS_SELECTOR, "button.shadow-button")
shadow_element.click()

```
**Notes:**
- Switching Frames: Ensure to switch back to the default content (driver.switch_to.default_content()) after interacting with elements inside iframes to avoid context issues in subsequent operations.

- Shadow DOM Limitations: Handling Shadow DOM requires JavaScript execution and can vary based on browser support and implementation details of the web application.


# Browser interactions
Get browser information
```
# Get Titile and URL
title = driver.title

url = driver.current_url
```
##  Browser navigation
### Navigate to
The first thing you will want to do after launching a browser is to open your website. This can be achieved in a single line:
```
driver.get("https://www.selenium.dev/selenium/web/index.html")

```
### Back
Pressing the browser’s back button:
```
driver.back()

```

### Refresh
Refresh the current page:

```
driver.refresh()

```

## Alerts
WebDriver can get the text from the popup and accept or dismiss these alerts. Can be closed these with `cancel` or `ok` button.

```
    element = driver.find_element(By.LINK_TEXT, "See an example alert")
    element.click()

    wait = WebDriverWait(driver, timeout=2)
    alert = wait.until(lambda d : d.switch_to.alert)
    alert.send_keys("Selenium") # if alter is prompt or editable
    text = alert.text
    alert.accept()

    alert.dismiss() # dismiss

```

## Working with Cookies
