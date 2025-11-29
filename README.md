# Build a Flight Booking AI Agent with Cycls

In this tutorial, we will develop and deploy a production-ready AI agent that acts as a flight booking assistant. Its main functions will be:

- Searching for flights using real-time data
- Displaying results in interactive cards
- Booking flights directly through the chat interface

The tech stack we’ll use:

- Cycls
- Python
- Docker
- APIs:
    - OpenAI API (LLM)
    - [Duffel](https://duffel.com/) API (for flight search and booking)

## Step 1: Build a simple agent

1.1 Install all dependencies

```bash
pip install cycls openai python-dotenv requests
```

1.2 Open Docker Desktop (to ensure it is started)

1.3 Create a new file called `agent.py`

```python
import cycls
agent = cycls.Agent() # Initialize the agent
@agent() # Decorate your function to register it as an agent
async def hello(context):
    yield "hi" # Your AI logic comes here
agent.deploy(prod=False) # Run your agent locally
```

1.4 Run script in terminal

```bash
python agent.py
```

1.5 Open your browser to [http://localhost:8080](http://localhost:8080)

## Step 2: Set up .env file and get API keys

Create a `.env` file and add API keys for:

- OpenAI
- Duffel (for flight data)
- Cycls

```env
OPENAI_API_KEY=sk-...
DUFFEL_API_KEY=duffel_test_...
CYCLS_API_KEY=...
```

## Step 3: Creating the tools for the agent to use

**Objective:** Write the Python functions that interact with the Duffel API to search and book flights.

The AI agent needs three capabilities:
- **Search:** Find flights based on origin, destination, and date.
- **Get Offer:** Retrieve details for a specific flight offer.
- **Create Order:** Book the flight.

Update your `agent.py` file with the following imports and helper functions.

```python
import cycls
import os
import requests
import urllib.parse
from openai import OpenAI
import dotenv
from datetime import datetime, timedelta

dotenv.load_dotenv()

# 1. Duffel API Wrapper
def duffel_request(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    headers = {
        "Authorization": f"Bearer {os.getenv('DUFFEL_API_KEY')}", 
        "Content-Type": "application/json", 
        "Duffel-Version": "v2"
    }
    try:
        if method == "POST":
            r = requests.post(f"https://api.duffel.com/{endpoint}", headers=headers, json=payload, timeout=30)
        else:
            r = requests.get(f"https://api.duffel.com/{endpoint}", headers=headers, timeout=30)
        
        if r.status_code >= 400:
            error_data = r.json() if r.content else {}
            return {"error": error_data.get("errors", [{"message": f"HTTP {r.status_code}: {r.text}"}])}
        
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# 2. Search Function
def search_flights(origin: str, destination: str, departure_date: str, passengers: int = 1):
    result = duffel_request("air/offer_requests", "POST", {
        "data": {
            "slices": [{"origin": origin, "destination": destination, "departure_date": departure_date}], 
            "passengers": [{"type": "adult"}] * passengers, 
            "cabin_class": "economy"
        }
    }) 
    
    if "error" in result or "errors" in result:
        # Error handling logic...
        errors = result.get('errors', [result.get('error')])
        return {"success": False, "error": f"❌ Error: {errors}"}
        
    offers = result.get("data", {}).get("offers", [])
    if not offers:
        return {"success": False, "error": "No flights found for your search criteria."}
    
    # Sort and slice offers (Prioritize Duffel Airways for testing)
    offers.sort(key=lambda x: 0 if x.get("owner", {}).get("name") == "Duffel Airways" else 1)
    offers = offers[:5]
    
    flights_data = []
    for offer in offers:
        flights_data.append({
            "offer_id": offer.get("id", ""),
            "airline": offer["owner"]["name"],
            "price": f"{offer['total_amount']} {offer['total_currency']}",
            "duration": offer["slices"][0]["duration"],
            "stops": len(offer["slices"][0].get("segments", [{}])) - 1,
            "departure": offer["slices"][0]["segments"][0].get("departing_at", "N/A").split("T")[1][:5],
            "arrival": offer["slices"][0]["segments"][-1].get("arriving_at", "N/A").split("T")[1][:5]
        })
    
    return {"success": True, "flights": flights_data, "origin": origin, "destination": destination}

# 3. Get Offer Details
def get_offer(offer_id: str):
    result = duffel_request(f"air/offers/{offer_id}", "GET")
    if "error" in result:
        return {"success": False, "error": str(result['error'])}
        
    offer_data = result.get("data", {})
    passenger_ids = [p.get("id") for p in offer_data.get("passengers", [])]
    
    return {
        "success": True, 
        "offer": offer_data, 
        "total_amount": offer_data.get("total_amount"), 
        "total_currency": offer_data.get("total_currency"),
        "passenger_ids": passenger_ids
    }

# 4. Create Booking Order
def create_order(offer_id: str, passengers: list, payment_type: str = "balance"):
    # First ensure we have latest price
    offer_result = get_offer(offer_id)
    if not offer_result.get("success"):
        return offer_result
        
    payload = {
        "data": {
            "selected_offers": [offer_id],
            "payments": [{
                "type": payment_type,
                "currency": offer_result["total_currency"],
                "amount": offer_result["total_amount"]
            }],
            "passengers": passengers
        }
    }
    
    result = duffel_request("air/orders", "POST", payload)
    if "error" in result:
        return {"success": False, "error": str(result['error'])}
    
    order_data = result.get("data", {})
    return {
        "success": True, 
        "booking_reference": order_data.get("booking_reference"), 
        "order_id": order_data.get("id")
    }
```

## Step 4: Adding the LLM logic

**Objective:** Connect the tools to the user via the Agent.

We will now rewrite the main agent function to define the tools and handle the conversation flow.

Update the `agent.py` file:

```python
# Initialize Agent with dependencies
agent = cycls.Agent(
    pip=["requests", "openai", "python-dotenv"], 
    copy=[".env"] 
)

@agent("flightagent")
async def flight_agent(context):
    dotenv.load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    # 1. Define System Persona
    messages = [{"role": "system", "content": f"""You are a helpful flight booking assistant.
        - Greet users warmly.
        - To search, ask for: origin, destination, and departure date.
        - Today is {today.strftime('%Y-%m-%d')}.
        - Use search_flights, get_offer, and create_order tools as needed.
        - When showing flights, provide the 'offer_id' so the user can book.
        - To book, collect passenger details (Name, DOB, Gender, Email, Phone).
        """}]
    messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in context.messages])
    
    # 2. Define Tools
    tools = [
        {"type": "function", "function": {"name": "search_flights", "description": "Search flights", "parameters": {
            "type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"}, 
                "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
                "passengers": {"type": "integer"}
            }, "required": ["origin", "destination", "departure_date"]
        }}},
        {"type": "function", "function": {"name": "get_offer", "description": "Get offer details", "parameters": {
            "type": "object", "properties": {"offer_id": {"type": "string"}}, "required": ["offer_id"]
        }}},
        {"type": "function", "function": {"name": "create_order", "description": "Book flight", "parameters": {
            "type": "object", "properties": {
                "offer_id": {"type": "string"},
                "passengers": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "given_name": {"type": "string"}, "family_name": {"type": "string"},
                    "gender": {"type": "string"}, "born_on": {"type": "string"}, "email": {"type": "string"}, 
                    "phone_number": {"type": "string"}
                }}}
            }, "required": ["offer_id", "passengers"]
        }}}
    ]

    # 3. Get LLM Response
    completion = openai_client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools, tool_choice="auto", temperature=0.7
    )
    response_msg = completion.choices[0].message
    
    # 4. Handle Tool Calls
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            if tool_call.function.name == "search_flights":
                args = json.loads(tool_call.function.arguments)
                result = search_flights(**args)
                # We will add UI rendering here in next step
                yield f"Found {len(result.get('flights', []))} flights."
            
            elif tool_call.function.name == "get_offer":
                # ... handle get_offer ...
                args = json.loads(tool_call.function.arguments)
                result = get_offer(args.get("offer_id"))
                yield f"Offer details: {result}"
                
            elif tool_call.function.name == "create_order":
                # ... handle create_order ...
                args = json.loads(tool_call.function.arguments)
                result = create_order(**args)
                if result.get("success"):
                    yield f"Booking Confirmed! Reference: {result.get('booking_reference')}"
                else:
                    yield f"Error: {result.get('error')}"
    else:
        yield response_msg.content
```

## Step 5: Add rich UI for flight results

Text-based flight lists are hard to read. We will inject HTML/CSS to display beautiful flight cards.

Update the `search_flights` handling block in `agent.py`:

```python
            if tool_call.function.name == "search_flights":
                args = json.loads(tool_call.function.arguments)
                result = search_flights(origin=args.get("origin"), destination=args.get("destination"), departure_date=args.get("departure_date"), passengers=args.get("passengers", 1))
                
                if result.get("success"):
                    flights = result.get("flights", [])
                    origin = result.get("origin")
                    destination = result.get("destination")
                    
                    # Container Styling
                    all_cards = '<style>@keyframes slideUpFade{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}} .flight-container { width: 100%; } @media (min-width: 768px) { .flight-container { min-width: 600px; } }</style>'
                    all_cards += '<div class="flight-container" style="min-height:600px; font-family:-apple-system,sans-serif;">'
                    all_cards += f'<div style="margin-bottom:24px; text-align:center;"><h2 style="margin:0; font-size:24px;">✈️ Flights to {destination}</h2><p>Found {len(flights)} options from {origin}</p></div>'
                    
                    for idx, flight in enumerate(flights, 1):
                        # Construct Booking URL
                        booking_message = f'Book Flight {idx} (ID: {flight.get("offer_id")}): {flight["airline"]}...'
                        booking_url = f"https://cycls.com/send/{urllib.parse.quote(booking_message)}"
                        
                        # Card HTML
                        all_cards += f'<div style="background:#fff; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.08); margin-bottom:24px; padding: 20px;">'
                        all_cards += f'<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f3f4f6; padding-bottom:16px;">'
                        all_cards += f'<h3 style="margin:0;">{flight["airline"]}</h3><div style="font-weight:800; font-size:20px;">{flight["price"]}</div></div>'
                        all_cards += f'<div style="padding:20px 0; display:flex; justify-content:space-between;"><div><div style="font-size:24px; font-weight:700;">{flight["departure"]}</div><div>{origin}</div></div>'
                        all_cards += f'<div><div style="font-size:24px; font-weight:700;">{flight["arrival"]}</div><div>{destination}</div></div></div>'
                        all_cards += f'<a href="{booking_url}" style="display:block; background:#111827; color:white; text-align:center; padding:12px; border-radius:8px; text-decoration:none;">Select Flight →</a>'
                        all_cards += '</div>'
                    
                    all_cards += '</div>'
                    yield all_cards
                else:
                    return f"<div style='color:red;'>{result.get('error')}</div>"
```

## Step 6: Add a header and intro section in UI

Create a `ui.py` file and add the following code for a professional travel look:

```python
header = """
<div class="fixed top-0 left-0 right-0 -z-40 h-[35vh] bg-cover bg-center" 
     style="background-image: url('https://images.unsplash.com/photo-1437846972679-9e6e537be46e?auto=format&fit=crop&q=80&w=2071');">
</div>
<div class="flex flex-col items-center justify-center text-center p-3 my-2 md:my-4">
  <div class="mb-32"></div>
  <div id="user-greeting" class="text-3xl sm:text-4xl font-bold text-gray-900 mb-2">Welcome to Flight Agent</div>
  <p class="text-gray-600 text-base sm:text-lg mt-1 mb-3">
    Your AI-powered assistant for finding and booking flights with real-time prices
  </p>
  <div class="flex justify-center items-center gap-2 mt-2">
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">Real-time prices</span>
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">AI-powered</span>
  </div>
</div>
"""

intro = """
<div class="py-1">
  <div class="flex flex-wrap gap-3 justify-center">
    <a href="https://cycls.com/send/${encodeURIComponent('Search flights from NYC to London tomorrow')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-white">
      <span>Search flights from NYC to London tomorrow</span>
    </a>
    <a href="https://cycls.com/send/${encodeURIComponent('Rechercher des vols Paris Rome')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-white">
      <span>Rechercher des vols Paris Rome</span>
    </a>
  </div>
</div>
"""
```

Now, update `agent.py` to use this UI:

1. Import the UI components:
```python
from ui import header, intro
```

2. Update the agent initialization to copy `ui.py`:
```python
agent = cycls.Agent(
    pip=["requests", "openai", "python-dotenv"], 
    copy=[".env", "ui.py"] 
)
```

3. Apply to the decorator:
```python
@agent("flightagent", header=header, intro=intro)
```

## Step 7: Add user authentication

To enable a sign-in/sign-up page for your agent, update the decorator:

```python
@agent("flightagent", header=header, intro=intro, auth=True)
```

## Step 8: Deploy to Cloud

Go from localhost to a public URL.

Update your agent initialization to include your Cycls API key:

```python
agent = cycls.Agent(
    key=os.getenv("CYCLS_API_KEY"), # Add your API Key
    pip=["requests", "openai", "python-dotenv"], 
    copy=[".env", "ui.py"]
)
```

Finally, change `prod=False` to `prod=True` in the last line:

```python
agent.deploy(prod=True)
```

Run your agent to deploy:

```bash
python agent.py
```

After a few minutes, you will receive a shareable public URL:
`🔗 Service is available at: https://flightagent-280879789566.me-central1.run.app`

## Step 9: Monitor your live AI Agent from Cycls Dashboard

Go to [https://cycls.com/dashboard](https://cycls.com/dashboard) to view all your live Agents in one dashboard.

# Congrats, you’ve just built and deployed your Flight Booking AI Agent!

---

**Complete `agent.py` file:**

```python
import cycls
import os
import requests
import urllib.parse
from openai import OpenAI
import dotenv
from datetime import datetime, timedelta
from ui import header, intro

dotenv.load_dotenv()

agent = cycls.Agent(
    key=os.getenv("CYCLS_API_KEY"),
    pip=["requests", "openai", "python-dotenv"], 
    copy=[".env", "ui.py"]
)

def duffel_request(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    headers = {"Authorization": f"Bearer {os.getenv('DUFFEL_API_KEY')}", "Content-Type": "application/json", "Duffel-Version": "v2"}
    try:
        if method == "POST":
            r = requests.post(f"https://api.duffel.com/{endpoint}", headers=headers, json=payload, timeout=30)
        else:
            r = requests.get(f"https://api.duffel.com/{endpoint}", headers=headers, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def search_flights(origin: str, destination: str, departure_date: str, passengers: int = 1):
    result = duffel_request("air/offer_requests", "POST", {"data": {"slices": [{"origin": origin, "destination": destination, "departure_date": departure_date}], "passengers": [{"type": "adult"}] * passengers, "cabin_class": "economy"}}) 
    if "error" in result or "errors" in result:
        return {"success": False, "error": f"Error: {result.get('error') or result.get('errors')}"}
        
    offers = result.get("data", {}).get("offers", [])
    if not offers: return {"success": False, "error": "No flights found."}
    
    offers.sort(key=lambda x: 0 if x.get("owner", {}).get("name") == "Duffel Airways" else 1)
    offers = offers[:5]
    
    flights_data = []
    for offer in offers:
        flights_data.append({
            "offer_id": offer.get("id", ""),
            "airline": offer["owner"]["name"],
            "price": f"{offer['total_amount']} {offer['total_currency']}",
            "duration": offer["slices"][0]["duration"],
            "stops": len(offer["slices"][0].get("segments", [{}])) - 1,
            "departure": offer["slices"][0]["segments"][0].get("departing_at", "N/A").split("T")[1][:5],
            "arrival": offer["slices"][0]["segments"][-1].get("arriving_at", "N/A").split("T")[1][:5]
        })
    return {"success": True, "flights": flights_data, "origin": origin, "destination": destination}

def get_offer(offer_id: str):
    result = duffel_request(f"air/offers/{offer_id}", "GET")
    if "error" in result: return {"success": False, "error": str(result)}
    offer_data = result.get("data", {})
    return {"success": True, "offer": offer_data, "total_amount": offer_data.get("total_amount"), "total_currency": offer_data.get("total_currency"), "passenger_ids": [p.get("id") for p in offer_data.get("passengers", [])]}

def create_order(offer_id: str, passengers: list, payment_type: str = "balance"):
    offer_res = get_offer(offer_id)
    if not offer_res.get("success"): return offer_res
    
    payload = {"data": {"selected_offers": [offer_id], "payments": [{"type": payment_type, "currency": offer_res["total_currency"], "amount": offer_res["total_amount"]}], "passengers": passengers}}
    result = duffel_request("air/orders", "POST", payload)
    if "error" in result: return {"success": False, "error": str(result)}
    return {"success": True, "booking_reference": result.get("data", {}).get("booking_reference")}

@agent("flightagent", header=header, intro=intro, auth=True)
async def flight_agent(context):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    messages = [{"role": "system", "content": f"You are a helpful flight booking assistant. Today is {today.strftime('%Y-%m-%d')}. Use search_flights, get_offer, and create_order tools."}]
    messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in context.messages])
    
    tools = [
        {"type": "function", "function": {"name": "search_flights", "description": "Search flights", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "departure_date": {"type": "string"}, "passengers": {"type": "integer"}}, "required": ["origin", "destination", "departure_date"]}}},
        {"type": "function", "function": {"name": "get_offer", "description": "Get offer", "parameters": {"type": "object", "properties": {"offer_id": {"type": "string"}}, "required": ["offer_id"]}}},
        {"type": "function", "function": {"name": "create_order", "description": "Book flight", "parameters": {"type": "object", "properties": {"offer_id": {"type": "string"}, "passengers": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "given_name": {"type": "string"}, "family_name": {"type": "string"}, "gender": {"type": "string"}, "born_on": {"type": "string"}, "email": {"type": "string"}, "phone_number": {"type": "string"}}}}}, "required": ["offer_id", "passengers"]}}}
    ]
    
    completion = openai_client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools, tool_choice="auto")
    response_msg = completion.choices[0].message
    
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            if tool_call.function.name == "search_flights":
                args = json.loads(tool_call.function.arguments)
                result = search_flights(**args)
                if result.get("success"):
                    flights = result.get("flights", [])
                    # ... (UI Generation Code Here) ...
                    all_cards = '<div class="flight-container">...</div>' # Simplified for brevity
                    yield all_cards
                else:
                    yield f"Error: {result.get('error')}"
            elif tool_call.function.name == "get_offer":
                args = json.loads(tool_call.function.arguments)
                yield str(get_offer(args.get("offer_id")))
            elif tool_call.function.name == "create_order":
                args = json.loads(tool_call.function.arguments)
                yield str(create_order(**args))
    else:
        yield response_msg.content

agent.deploy(False)
```
