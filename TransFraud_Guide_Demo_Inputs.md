# TransFraud --- Guide Demonstration Input Sheet

**Project:** TransFraud --- A Real-Time Transformer-Based Model for
Financial Fraud Detection\
**Purpose:** Sample inputs to demonstrate how the application accepts a
payment, evaluates risk, and displays a decision.

> **Before the demo:** Start the FastAPI backend and React frontend. Use
> synthetic/demo identifiers only. The model's probability and risk
> result can depend on transaction history, device history, timestamps,
> preprocessing, and the current model. Record the actual output shown
> during your run.

## 1. Start the application

-   Start the backend using the project's configured Uvicorn command.
-   Open the frontend in your browser.
-   Confirm the API documentation is available at
    `http://127.0.0.1:8000/docs`.
-   Keep the backend terminal visible if your guide wants to see API
    requests.

## 2. Inputs to enter

Use these as demonstration examples only. If an identifier is not
present in the dataset/history loaded by your current application,
substitute one you have verified locally.

### Demo A --- Existing customer and known device

  Field         Example input
  ------------- -----------------
  Customer ID   `CUST0000186`
  Receiver ID   `TEST001`
  Amount        `100`
  Card ID       `CARD0000186-3`
  Device ID     `4b00bd1c7e06`
  Merchant ID   `MID017968`

**What to demonstrate:** Submit the payment and point out the
probability, risk level, and decision returned by the system.

**What to explain:** The application combines transaction details with
available customer/card/device/merchant history to build model inputs. A
familiar device may produce a different assessment from an unfamiliar
one, but a low-risk result is not guaranteed.

### Demo B --- Same customer, unfamiliar device

Keep the values from Demo A and change only this field:

  Field       Example input
  ----------- ----------------------------
  Device ID   `BRAND_NEW_DEVICE_TEST999`

**What to demonstrate:** Submit the payment again and compare the
assessment with Demo A.

**What to explain:** The backend derives a new-device/history feature
and sends the processed inputs to the trained model. A new device can
increase the risk score in this example, but the result depends on the
current model and feature pipeline.

### Demo C --- Built-in fraud demonstration

Use the application's **Demo Fraud Case** button, if available.

**What to demonstrate:** Show the returned probability, risk level, and
recommended decision.

**What to explain:** This is a predefined demonstration scenario
designed to show how the interface presents a high-risk assessment. Do
not claim that every real fraudulent payment will be detected.

## 3. Validation demonstrations

  -----------------------------------------------------------------------
  Test                    Input/action            Expected application
                                                  behavior
  ----------------------- ----------------------- -----------------------
  Missing customer        Clear Customer ID and   Validation message;
                          submit                  request should not be
                                                  submitted

  Missing amount          Leave Amount blank and  Validation message
                          submit                  

  Zero amount             Enter `0` and submit    Amount validation
                                                  message

  Negative amount         Enter `-100` and submit Amount validation
                                                  message

  Backend unavailable     Stop backend and submit Clear error message; no
                          a valid form            false successful result
  -----------------------------------------------------------------------

## 4. Explain the output

When the result appears, point out:

-   **Fraud probability:** The model's estimated fraud score/probability
    for this input. It is not a guarantee.
-   **Risk level:** The application's category derived from its
    configured thresholds.
-   **Decision:** The application's recommended action, such as Proceed,
    Verify, or Alert / Verify.
-   **Feature/history effect:** Customer, card, device, merchant,
    amount, and time-related features can influence the assessment.

For the current backend policy, the configured categories are: -
Probability below `0.50`: Low - Probability from `0.50` to below `0.95`:
Medium - Probability `0.95` or above: High

Current decisions: - Low → Proceed - Medium → Verify - High → Alert /
Verify

Confirm these values against the running `app.py` before presenting if
you have changed the risk policy.

## 5. Simple explanation to say to your guide

"First, I enter the payment details in the React frontend. The FastAPI
backend validates the request and derives features using the available
transaction history, including customer, card, device, and merchant
behavior. The preprocessing pipeline prepares the model inputs, and the
trained BiLSTM--Transformer model returns a fraud score. The backend
maps that score to a risk category and decision, which the frontend
displays. This is a decision-support prototype, so its output depends on
the trained model and available features."

## 6. Record the actual demo results

Fill this in after you run the application:

  Scenario                  Actual probability   Risk   Decision   Pass/Fail
  ------------------------- -------------------- ------ ---------- -----------
  Demo A --- Known device                                          
  Demo B --- New device                                            
  Demo C --- Fraud demo                                            

## Important notes

-   Do not enter real customer, card, or banking information in a
    college demo.
-   Do not promise a particular score unless you have just run the same
    scenario and verified it.
-   A new device is a risk signal, not proof of fraud.
-   Explain only the model architecture and preprocessing actually
    implemented in your project.
