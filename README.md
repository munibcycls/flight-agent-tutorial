# Tutorial - Flight Agent

In this tutorial, we will develop and deploy a production-ready AI agent that acts as a flight booking assistant. Its main functions will be:

- Searching for flights using real-time data
- Displaying results in interactive cards
- Booking flights directly through the chat interface

The tech stack we’ll use:

- Cycls ([https://cycls.com](https://cycls.com/))
- Python
- Docker ([https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/))
- APIs:
    - OpenAI API (LLM)
    - Duffel API (for real time flight search) ([https://duffel.com](https://duffel.com/))

# Step 1: Build a simple agent

1.1 Install all dependencies

```basic
pip install cycls openai python-dotenv requests
```

1.2 Open docker (to ensure it is started)

1.3 Create a new file called agent.py

```basic
import cycls

agent = cycls.Agent() # Initialize the agent

@agent() # Decorate your function to register it as an agent
async def hello(context):
    yield "hi" #Your AI logic comes here

agent.deploy(prod=False) # Run your agent locally
```

1.4 Run script in terminal

```basic
python agent.py
```

1.5 Open your browser to [http://localhost:8080](http://127.0.0.1:8000/)

# Step 2: Set up .env file and get API keys

Create .env file and add API keys for:

- OpenAI
- Duffel
- Cycls

```
OPENAI_API_KEY=sk-...
DUFFEL_API_KEY=duffel_test_...
CYCLS_API_KEY=...
```

# Step 3: Creating the tools for the agent to use

**Objective:** Write the Python functions that interact with the Duffel API to search and book flights.

Write 4 helper functions (tools):

- Duffel API Wrapper
- Search Flights
- Get offer details
- Create Booking Order

Update your `agent.py` file with the following imports and helper functions.

```basic
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

# Step 4: Adding the LLM logic & Streaming UI

**Objective:** Connect the tools to the user via the Agent.

We will now rewrite the main agent function to define the tools and handle the conversation flow.

Update the `agent.py` file:

```basic
@agent("flightagent")
async def flight_agent(context):
    import os
    from openai import OpenAI
    import dotenv
    from datetime import datetime, timedelta
    import json
    dotenv.load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  
    today = datetime.now()
    tomorrow = today + timedelta(days=1)  
    
    messages = [{"role": "system", "content": f"""You are a helpful flight booking assistant.
        Your job is to help users find and book flights.
        - Greet users warmly and ask how you can help with their travel plans
        - When they want to search flights, ask for: origin, destination, and departure date
        - IMPORTANT: Today is {today.strftime('%Y-%m-%d')}. When user says "tomorrow", use {tomorrow.strftime('%Y-%m-%d')}
        - Departure dates must be {tomorrow.strftime('%Y-%m-%d')} or later (no same-day bookings)
        - Once you have all details, use the search_flights tool. 
        - NOTE: If using a test API key, "Duffel Airways" is the most reliable airline for booking testing. Recommend it if available.
        - When user selects a flight (via "Book Flight X" message), the message will contain the Offer ID (e.g., "ID: off_..."). Extract this ID.
        - First use get_offer to retrieve the latest offer AND the valid passenger_ids.
        - Collect passenger details: given_name, family_name, email, phone_number, born_on (YYYY-MM-DD), gender (m/f), title (mr/mrs/ms/miss)
        - IMPORTANT: Phone numbers MUST be in E.164 format (e.g., +14155552671). Ask user for country code if missing.
        - Use the passenger_ids returned by get_offer - map them in order (first ID for first passenger, etc.). Do NOT make up IDs.
        - Once you have all passenger details, use create_order to complete the booking
        - Display booking confirmation with booking_reference when order is created successfully
        - Be conversational and friendly throughout"""}]
    messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in context.messages])
    
    tools = [
        {"type": "function", "function": {"name": "search_flights", "description": "Search for flights between two airports on a specific date", "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "Origin airport code (e.g., 'JFK', 'CAI')"}, "destination": {"type": "string", "description": "Destination airport code (e.g., 'LAX', 'JFK')"}, "departure_date": {"type": "string", "description": "Date in YYYY-MM-DD format (must be tomorrow or later)"}, "passengers": {"type": "integer", "description": "Number of passengers", "default": 1}}, "required": ["origin", "destination", "departure_date"]}}},
        {"type": "function", "function": {"name": "get_offer", "description": "Retrieve the latest version of an offer to get up-to-date pricing and passenger IDs before booking", "parameters": {"type": "object", "properties": {"offer_id": {"type": "string", "description": "The offer ID from search results"}}, "required": ["offer_id"]}}},
        {"type": "function", "function": {"name": "create_order", "description": "Create a booking order for a selected flight offer", "parameters": {"type": "object", "properties": {"offer_id": {"type": "string", "description": "The offer ID to book"}, "passengers": {"type": "array", "description": "Array of passenger objects", "items": {"type": "object", "properties": {"id": {"type": "string", "description": "The passenger ID from the get_offer response"}, "given_name": {"type": "string"}, "family_name": {"type": "string"}, "gender": {"type": "string", "enum": ["m", "f"]}, "title": {"type": "string", "enum": ["mr", "ms", "mrs", "miss", "dr"]}, "born_on": {"type": "string", "description": "YYYY-MM-DD"}, "email": {"type": "string"}, "phone_number": {"type": "string", "description": "E.164 format (e.g. +14155552671)"}}, "required": ["id", "given_name", "family_name", "gender", "born_on", "email", "phone_number"]}}, "payment_type": {"type": "string", "description": "Payment type: 'balance' or 'arc_bsp_cash'", "default": "balance"}, "total_amount": {"type": "number", "description": "Total amount from offer"}, "total_currency": {"type": "string", "description": "Currency code"}}, "required": ["offer_id", "passengers"]}}}
    ]
    
    completion = openai_client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools, tool_choice="auto", temperature=0.7)
    response_msg = completion.choices[0].message
    
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            if tool_call.function.name == "search_flights":
                args = json.loads(tool_call.function.arguments)
                result = search_flights(origin=args.get("origin"), destination=args.get("destination"), departure_date=args.get("departure_date"), passengers=args.get("passengers", 1))
                
                if result.get("success"):
                    flights = result.get("flights", [])
                    origin = result.get("origin")
                    destination = result.get("destination")
                    
                    # Main container styling: Fixed height wrapper to contain layout during streaming
                    # Includes animation styles for staggered card entry
                    all_cards = '<style>@keyframes slideUpFade{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}} .flight-container { width: 100%; } @media (min-width: 768px) { .flight-container { min-width: 600px; } }</style>'
                    all_cards += '<div class="flight-container" style="min-height:600px; box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">'
                    all_cards += f'<div style="margin-bottom:24px; text-align:center; animation:slideUpFade 0.5s ease-out forwards;"><h2 style="margin:0; font-size:24px; font-weight:700; color:#111827;">✈️ Flights to {destination}</h2><p style="margin:8px 0 0 0; color:#6b7280; font-size:14px;">Found {len(flights)} options from {origin}</p></div>'
                    
                    for idx, flight in enumerate(flights, 1):
                        price_parts = flight['price'].split()
                        price_amount = price_parts[0]
                        price_currency = price_parts[1] if len(price_parts) > 1 else ''
                        
                        # Logic for stops display
                        is_direct = flight['stops'] == 0
                        stops_text = 'Direct' if is_direct else f"{flight['stops']} Stop{'s' if flight['stops'] > 1 else ''}"
                        stops_color = '#10b981' if is_direct else '#f59e0b' # Green for direct, Amber for stops
                        
                        # Booking URL construction
                        offer_id = flight.get('offer_id', '')
                        airline = flight['airline']
                        booking_message = f'Book Flight {idx} (ID: {offer_id}): {airline} from {origin} to {destination} at {price_amount} {price_currency}'
                        booking_url = f"https://cycls.com/send/{urllib.parse.quote(booking_message)}"
                        
                        # Animation delay calculation
                        anim_delay = (idx - 1) * 0.15
                        
                        # Card HTML
                        all_cards += f'<div style="background:#ffffff; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.08); margin-bottom:24px; overflow:hidden; width:100%; border:1px solid #f3f4f6; opacity:0; animation:slideUpFade 0.5s ease-out {anim_delay}s forwards;">'
                        
                        # Header: Airline & Price
                        all_cards += f'<div style="padding: 20px 24px; border-bottom: 1px solid #f3f4f6; display: flex; justify-content: space-between; align-items: center; background: #ffffff;">'
                        all_cards += f'<div style="display: flex; align-items: center; gap: 12px;">'
                        all_cards += f'<div style="width: 40px; height: 40px; background: #f3f4f6; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #4b5563;"><svg style="width:20px;height:20px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg></div>'
                        all_cards += f'<div><h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #111827;">{flight["airline"]}</h3><div style="font-size: 12px; color: #6b7280; margin-top: 2px;">Flight {idx}</div></div>'
                        all_cards += f'</div><div style="text-align: right;"><div style="font-size: 20px; font-weight: 800; color: #111827;">{price_amount} <span style="font-size: 14px; font-weight: 500; color: #6b7280;">{price_currency}</span></div></div></div>'
                        
                        # Body: Times & Route
                        all_cards += f'<div style="padding: 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px;">'
                        all_cards += f'<div style="text-align: left; flex: 1;"><div style="font-size: 24px; font-weight: 700; color: #111827; line-height: 1.2;">{flight["departure"]}</div><div style="font-size: 14px; font-weight: 600; color: #9ca3af; margin-top: 4px;">{origin}</div></div>'
                        all_cards += f'<div style="flex: 2; display: flex; flex-direction: column; align-items: center; position: relative; padding: 0 10px;"><div style="font-size: 12px; font-weight: 600; color: #6b7280; margin-bottom: 8px;">{flight["duration"]}</div>'
                        all_cards += f'<div style="width: 100%; height: 2px; background: #e5e7eb; position: relative; border-radius: 2px;"><div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: {stops_color}; padding: 0 8px; font-size: 10px; font-weight: 700; color: white; border-radius: 10px; line-height: 16px; white-space: nowrap;">{stops_text}</div></div></div>'
                        all_cards += f'<div style="text-align: right; flex: 1;"><div style="font-size: 24px; font-weight: 700; color: #111827; line-height: 1.2;">{flight["arrival"]}</div><div style="font-size: 14px; font-weight: 600; color: #9ca3af; margin-top: 4px;">{destination}</div></div></div>'
                        
                        # Footer: Action
                        all_cards += f'<div style="padding: 16px 24px; background: #f9fafb; border-top: 1px solid #f3f4f6; display: flex; align-items: center; justify-content: space-between;">'
                        all_cards += f'<div style="display: flex; gap: 16px; font-size: 12px; font-weight: 500; color: #6b7280;"><span style="display: flex; align-items: center; gap: 4px;">🧳 Included</span><span style="display: flex; align-items: center; gap: 4px;">💺 Economy</span></div>'
                        all_cards += f'<a href="{booking_url}" style="background: #111827; color: white; text-decoration: none; padding: 12px 28px; border-radius: 8px; font-weight: 600; font-size: 14px; transition: all 0.2s; display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">Select Flight →</a></div>'
                        
                        all_cards += '</div>'
                    
                    all_cards += '<div style="text-align:center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;"><p style="color: #9ca3af; font-size: 12px; margin: 0;">Prices include all taxes and fees • 24/7 Support</p></div></div>'
                    return all_cards
                else:
                    return f"<div style='padding: 20px; color: red; background: #fee; border-radius: 8px; font-family: sans-serif;'>{result.get('error')}</div>"
            elif tool_call.function.name == "get_offer":
                args = json.loads(tool_call.function.arguments)
                result = get_offer(offer_id=args.get("offer_id"))
                if result.get("success"):
                    offer = result.get("offer", {})
                    total_amount = result.get("total_amount")
                    total_currency = result.get("total_currency")
                    passenger_ids = result.get("passenger_ids", [])
                    return f"✅ Offer retrieved successfully. Current price: {total_amount} {total_currency}.\n\nIMPORTANT for Agent: You MUST use these Passenger IDs for the booking: {passenger_ids}\n\nPlease collect passenger details (Name, DOB, Gender, Email, Phone + Country Code)."
                elif result.get("expired"):
                    return f"⚠️ {result.get('error')} You can still proceed with booking using the original offer price."
                else:
                    return f"<div style='padding: 20px; color: red; background: #fee; border-radius: 8px; font-family: sans-serif;'>{result.get('error')}</div>"
            elif tool_call.function.name == "create_order":
                args = json.loads(tool_call.function.arguments)
                result = create_order(
                    offer_id=args.get("offer_id"), 
                    passengers=args.get("passengers"), 
                    payment_type=args.get("payment_type", "balance"),
                    total_amount=args.get("total_amount"),
                    total_currency=args.get("total_currency")
                )
                if result.get("success"):
                    booking_ref = result.get("booking_reference")
                    order_id = result.get("order_id")
                    return f"""<div style='padding: 24px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; color: white; font-family: sans-serif;'>
                        <h2 style='margin: 0 0 16px 0; font-size: 24px;'>🎉 Booking Confirmed!</h2>
                        <p style='margin: 8px 0; font-size: 16px;'><strong>Booking Reference:</strong> {booking_ref}</p>
                        <p style='margin: 8px 0; font-size: 16px;'><strong>Order ID:</strong> {order_id}</p>
                        <p style='margin: 16px 0 0 0; font-size: 14px; opacity: 0.9;'>Your flight has been successfully booked. You can use the booking reference to check your reservation on the airline's website.</p>
                    </div>"""
                else:
                    return f"<div style='padding: 20px; color: red; background: #fee; border-radius: 8px; font-family: sans-serif;'>{result.get('error')}</div>"
    
    return response_msg.content or "Hello! I'm your flight booking assistant. Where would you like to fly today?"
```

# Step 5: Test Locally

```
agent.deploy(prod=True)
```

# Step 6: Add a header and intro section in UI

Create a `ui.py` file and add the following code for a professional travel look:

```basic
header = """
<div class="fixed top-0 left-0 right-0 -z-40 h-[35vh] bg-cover bg-center bg-no-repeat" 
     style="background-image: url('https://images.unsplash.com/photo-1437846972679-9e6e537be46e?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=2071');
            -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0.95) 20%, rgba(0,0,0,0.8) 40%, rgba(0,0,0,0.6) 60%, rgba(0,0,0,0.3) 80%, rgba(0,0,0,0.1) 90%, rgba(0,0,0,0) 100%);
            mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 0%, rgba(0,0,0,0.95) 20%, rgba(0,0,0,0.8) 40%, rgba(0,0,0,0.6) 60%, rgba(0,0,0,0.3) 80%, rgba(0,0,0,0.1) 90%, rgba(0,0,0,0) 100%);">
</div>
<div class="flex flex-col items-center justify-center text-center p-3 my-2 md:my-4">
  <div class="mb-32">
  </div>
  <div id="user-greeting" class="text-3xl sm:text-4xl font-bold text-gray-900 mb-2">Welcome to Flight Agent</div>
  <p class="text-gray-600 text-base sm:text-lg mt-1 mb-3">
    Your AI-powered assistant for finding and booking flights with real-time prices
  </p>
  <div class="flex justify-center items-center gap-2 mt-2 mb-2 flex-wrap">
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">Real-time prices</span>
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">Live flight data</span>
    <span class="px-2 py-1 bg-gray-50 text-gray-700 rounded-full text-xs">AI-powered</span>
  </div>
  <div class="flex justify-center items-center mt-2">
    <a href="https://github.com/Cycls/Flight-Agent" target="_blank" rel="noopener noreferrer" class="flex items-center gap-2 px-3 py-1 bg-black text-white rounded-lg hover:bg-gray-800 transition-all no-underline">
      <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
      </svg>
      <span class="text-sm font-medium">GitHub</span>
    </a>
  </div>
</div>
"""

intro = """
---

<div class="py-1">
  <div class="flex flex-wrap gap-3 justify-center">
    <a href="https://cycls.com/send/${encodeURIComponent('Search flights from NYC to London tomorrow')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>Search flights from NYC to London tomorrow</span>
    </a>
    <a href="https://cycls.com/send/${encodeURIComponent('Buscar vuelos de Madrid a Barcelona')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>Buscar vuelos de Madrid a Barcelona</span>
    </a>
    <a href="https://cycls.com/send/${encodeURIComponent('Rechercher des vols Paris Rome')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>Rechercher des vols Paris Rome</span>
    </a>
    <a href="https://cycls.com/send/${encodeURIComponent('東京からシンガポールへのフライトを検索')}" class="group relative inline-flex items-center justify-center px-4 py-2 overflow-hidden font-medium text-gray-700 border-2 border-gray-300 rounded-xl shadow-lg bg-gradient-to-br from-gray-50 to-white focus:outline-none hover:border-gray-400 hover:shadow-xl transition-all whitespace-nowrap text-sm">
      <span>東京からシンガポールへのフライトを検索</span>
    </a>
  </div>
</div>

"""
```

Now, update `agent.py` to use this UI:

Import the UI components:

```python
from ui import header, intro
```

Update the agent initialization to copy `ui.py`:

```python
agent = cycls.Agent(
    pip=["requests", "openai", "python-dotenv"],
    copy=[".env", "ui.py"]
)
```

Apply to the decorator:

```python
@agent("flightagent", header=header, intro=intro)
```

# Step 7: Add user authentication

To enable a sign-in/sign-up page for your agent, update the decorator:

```python
@agent("flightagent", header=header, intro=intro, auth=True)
```

# Step 8: Deploy to Cloud

Go from localhost to a public URL.

Update your agent initialization to include your Cycls API key:

```basic
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
`🔗 Service is available at: https://flightagent.cycls.ai`

# Step 9: Monitor your live AI Agent from Cycls Dashboard

Go to [https://cycls.com/dashboard](https://cycls.com/dashboard) to view all your live Agents in one dashboard.

# Congrats, you’ve just built and deployed your Flight Booking AI Agent!

Here is the complete [agent.py](http://agent.py) file:

```basic
import cycls
import os
import requests
import urllib.parse
from openai import OpenAI
import dotenv
from datetime import datetime, timedelta
from ui import header, intro

dotenv.load_dotenv()

agent = cycls.Agent(key = os.getenv("CYCLS_API_KEY"),
 pip=["requests", "openai", "python-dotenv"], copy=[".env"])

def duffel_request(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    headers = {"Authorization": f"Bearer {os.getenv('DUFFEL_API_KEY')}", "Content-Type": "application/json", "Duffel-Version": "v2"}
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

def search_flights(origin: str, destination: str, departure_date: str, passengers: int = 1):
    result = duffel_request("air/offer_requests", "POST", {"data": {"slices": [{"origin": origin, "destination": destination, "departure_date": departure_date}], "passengers": [{"type": "adult"}] * passengers, "cabin_class": "economy"}}) 
    if "error" in result or "errors" in result:
        errors = result.get('errors', [result.get('error')])
        if isinstance(errors, list) and len(errors) > 0:
            error_msg = errors[0].get('message', str(errors[0])) if isinstance(errors[0], dict) else str(errors[0])
            return {"success": False, "error": "❌ Sorry, the departure date must be in the future. Please choose a date starting from tomorrow or later."} if 'must be after' in error_msg else {"success": False, "error": f"❌ {error_msg}"}
        return {"success": False, "error": f"❌ Error: {errors}"}
    offers = result.get("data", {}).get("offers", [])
    if not offers:
        return {"success": False, "error": "No flights found for your search criteria."}
    
    # Sort offers to prioritize Duffel Airways (ZZ) for reliable testing
    # Duffel Airways usually has 'owner' -> 'iata_code' = 'ZZ' or 'name' = 'Duffel Airways'
    offers.sort(key=lambda x: 0 if x.get("owner", {}).get("name") == "Duffel Airways" or x.get("owner", {}).get("iata_code") == "ZZ" else 1)
    
    offers = offers[:5] # Take top 5 after sorting
    
    offer_request_data = result.get("data", {})
    passenger_ids = [p.get("id") for p in offer_request_data.get("passengers", [])]
    offer_request_id = offer_request_data.get("id", "")
    
    flights_data = []
    for offer in offers:
        flights_data.append({
            "offer_id": offer.get("id", ""),
            "airline": offer["owner"]["name"],
            "price": f"{offer['total_amount']} {offer['total_currency']}",
            "total_amount": offer.get("total_amount"),
            "total_currency": offer.get("total_currency"),
            "duration": offer["slices"][0]["duration"],
            "stops": len(offer["slices"][0].get("segments", [{}])) - 1,
            "departure": offer["slices"][0]["segments"][0].get("departing_at", "N/A").split("T")[1][:5] if "T" in offer["slices"][0]["segments"][0].get("departing_at", "") else "N/A",
            "arrival": offer["slices"][0]["segments"][-1].get("arriving_at", "N/A").split("T")[1][:5] if "T" in offer["slices"][0]["segments"][-1].get("arriving_at", "") else "N/A"
        })
    
    return {"success": True, "flights": flights_data, "origin": origin, "destination": destination, "passenger_ids": passenger_ids, "offer_request_id": offer_request_id}

def get_offer(offer_id: str):
    result = duffel_request(f"air/offers/{offer_id}", "GET")
    if "error" in result or "errors" in result:
        errors = result.get('errors', [result.get('error')])
        if isinstance(errors, list) and len(errors) > 0:
            error_msg = errors[0].get('message', str(errors[0])) if isinstance(errors[0], dict) else str(errors[0])
            # If offer doesn't exist, it might have expired - this is okay, we can still try to create order
            if "does not exist" in error_msg.lower() or "not found" in error_msg.lower():
                return {"success": False, "error": f"❌ Offer may have expired. Error: {error_msg}", "expired": True}
            return {"success": False, "error": f"❌ {error_msg}"}
        return {"success": False, "error": f"❌ Error: {errors}"}
    offer_data = result.get("data", {})
    if not offer_data:
        return {"success": False, "error": "❌ No offer data returned", "expired": True}
    
    # Extract passenger IDs from the offer to ensure we use the correct ones for booking
    passenger_ids = [p.get("id") for p in offer_data.get("passengers", [])]
    
    return {
        "success": True, 
        "offer": offer_data, 
        "total_amount": offer_data.get("total_amount"), 
        "total_currency": offer_data.get("total_currency"),
        "passenger_ids": passenger_ids
    }

def create_order(offer_id: str, passengers: list, payment_type: str = "balance", total_amount: float = None, total_currency: str = None):
    # Try to get the latest offer to ensure price is up-to-date
    # If it fails (offer expired), use provided amounts as fallback
    offer_result = get_offer(offer_id)
    if offer_result.get("success"):
        total_amount = offer_result.get("total_amount")
        total_currency = offer_result.get("total_currency")
    elif offer_result.get("expired") and (total_amount is None or total_currency is None):
        # If get_offer failed because offer expired and we don't have fallback amounts, return error
        return {"success": False, "error": "❌ This offer has expired. Please search for flights again to get a fresh offer."}
    
    payload = {
        "data": {
            "selected_offers": [offer_id],
            "payments": [{
                "type": payment_type,
                "currency": total_currency,
                "amount": total_amount
            }],
            "passengers": passengers
        }
    }
    
    result = duffel_request("air/orders", "POST", payload)
    if "error" in result or "errors" in result:
        errors = result.get('errors', [result.get('error')])
        if isinstance(errors, list) and len(errors) > 0:
            error_obj = errors[0]
            error_msg = error_obj.get('message', str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
            
            # Specific handling for expired offers in test mode
            if isinstance(error_obj, dict) and error_obj.get('code') == 'offer_no_longer_available':
                return {"success": False, "error": "❌ This flight offer has expired or is no longer available. In Test Mode, please try booking a 'Duffel Airways' flight for guaranteed success."}
                
            return {"success": False, "error": f"❌ {error_msg}"}
        return {"success": False, "error": f"❌ Error: {errors}"}
    
    order_data = result.get("data", {})
    return {"success": True, "order": order_data, "booking_reference": order_data.get("booking_reference"), "order_id": order_data.get("id")}

@agent("flightagent", header=header, intro=intro)

async def flight_agent(context):
    import os
    from openai import OpenAI
    import dotenv
    from datetime import datetime, timedelta
    import json
    dotenv.load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  
    today = datetime.now()
    tomorrow = today + timedelta(days=1)  
    
    messages = [{"role": "system", "content": f"""You are a helpful flight booking assistant.
        Your job is to help users find and book flights.
        - Greet users warmly and ask how you can help with their travel plans
        - When they want to search flights, ask for: origin, destination, and departure date
        - IMPORTANT: Today is {today.strftime('%Y-%m-%d')}. When user says "tomorrow", use {tomorrow.strftime('%Y-%m-%d')}
        - Departure dates must be {tomorrow.strftime('%Y-%m-%d')} or later (no same-day bookings)
        - Once you have all details, use the search_flights tool. 
        - NOTE: If using a test API key, "Duffel Airways" is the most reliable airline for booking testing. Recommend it if available.
        - When user selects a flight (via "Book Flight X" message), the message will contain the Offer ID (e.g., "ID: off_..."). Extract this ID.
        - First use get_offer to retrieve the latest offer AND the valid passenger_ids.
        - Collect passenger details: given_name, family_name, email, phone_number, born_on (YYYY-MM-DD), gender (m/f), title (mr/mrs/ms/miss)
        - IMPORTANT: Phone numbers MUST be in E.164 format (e.g., +14155552671). Ask user for country code if missing.
        - Use the passenger_ids returned by get_offer - map them in order (first ID for first passenger, etc.). Do NOT make up IDs.
        - Once you have all passenger details, use create_order to complete the booking
        - Display booking confirmation with booking_reference when order is created successfully
        - Be conversational and friendly throughout"""}]
    messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in context.messages])
    
    tools = [
        {"type": "function", "function": {"name": "search_flights", "description": "Search for flights between two airports on a specific date", "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "Origin airport code (e.g., 'JFK', 'CAI')"}, "destination": {"type": "string", "description": "Destination airport code (e.g., 'LAX', 'JFK')"}, "departure_date": {"type": "string", "description": "Date in YYYY-MM-DD format (must be tomorrow or later)"}, "passengers": {"type": "integer", "description": "Number of passengers", "default": 1}}, "required": ["origin", "destination", "departure_date"]}}},
        {"type": "function", "function": {"name": "get_offer", "description": "Retrieve the latest version of an offer to get up-to-date pricing and passenger IDs before booking", "parameters": {"type": "object", "properties": {"offer_id": {"type": "string", "description": "The offer ID from search results"}}, "required": ["offer_id"]}}},
        {"type": "function", "function": {"name": "create_order", "description": "Create a booking order for a selected flight offer", "parameters": {"type": "object", "properties": {"offer_id": {"type": "string", "description": "The offer ID to book"}, "passengers": {"type": "array", "description": "Array of passenger objects", "items": {"type": "object", "properties": {"id": {"type": "string", "description": "The passenger ID from the get_offer response"}, "given_name": {"type": "string"}, "family_name": {"type": "string"}, "gender": {"type": "string", "enum": ["m", "f"]}, "title": {"type": "string", "enum": ["mr", "ms", "mrs", "miss", "dr"]}, "born_on": {"type": "string", "description": "YYYY-MM-DD"}, "email": {"type": "string"}, "phone_number": {"type": "string", "description": "E.164 format (e.g. +14155552671)"}}, "required": ["id", "given_name", "family_name", "gender", "born_on", "email", "phone_number"]}}, "payment_type": {"type": "string", "description": "Payment type: 'balance' or 'arc_bsp_cash'", "default": "balance"}, "total_amount": {"type": "number", "description": "Total amount from offer"}, "total_currency": {"type": "string", "description": "Currency code"}}, "required": ["offer_id", "passengers"]}}}
    ]
    
    completion = openai_client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools, tool_choice="auto", temperature=0.7)
    response_msg = completion.choices[0].message
    
    if response_msg.tool_calls:
        for tool_call in response_msg.tool_calls:
            if tool_call.function.name == "search_flights":
                args = json.loads(tool_call.function.arguments)
                result = search_flights(origin=args.get("origin"), destination=args.get("destination"), departure_date=args.get("departure_date"), passengers=args.get("passengers", 1))
                
                if result.get("success"):
                    flights = result.get("flights", [])
                    origin = result.get("origin")
                    destination = result.get("destination")
                    
                    # Main container styling: Fixed height wrapper to contain layout during streaming
                    # Includes animation styles for staggered card entry
                    all_cards = '<style>@keyframes slideUpFade{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}} .flight-container { width: 100%; } @media (min-width: 768px) { .flight-container { min-width: 600px; } }</style>'
                    all_cards += '<div class="flight-container" style="min-height:600px; box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">'
                    all_cards += f'<div style="margin-bottom:24px; text-align:center; animation:slideUpFade 0.5s ease-out forwards;"><h2 style="margin:0; font-size:24px; font-weight:700; color:#111827;">✈️ Flights to {destination}</h2><p style="margin:8px 0 0 0; color:#6b7280; font-size:14px;">Found {len(flights)} options from {origin}</p></div>'
                    
                    for idx, flight in enumerate(flights, 1):
                        price_parts = flight['price'].split()
                        price_amount = price_parts[0]
                        price_currency = price_parts[1] if len(price_parts) > 1 else ''
                        
                        # Logic for stops display
                        is_direct = flight['stops'] == 0
                        stops_text = 'Direct' if is_direct else f"{flight['stops']} Stop{'s' if flight['stops'] > 1 else ''}"
                        stops_color = '#10b981' if is_direct else '#f59e0b' # Green for direct, Amber for stops
                        
                        # Booking URL construction
                        offer_id = flight.get('offer_id', '')
                        airline = flight['airline']
                        booking_message = f'Book Flight {idx} (ID: {offer_id}): {airline} from {origin} to {destination} at {price_amount} {price_currency}'
                        booking_url = f"https://cycls.com/send/{urllib.parse.quote(booking_message)}"
                        
                        # Animation delay calculation
                        anim_delay = (idx - 1) * 0.15
                        
                        # Card HTML
                        all_cards += f'<div style="background:#ffffff; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,0.08); margin-bottom:24px; overflow:hidden; width:100%; border:1px solid #f3f4f6; opacity:0; animation:slideUpFade 0.5s ease-out {anim_delay}s forwards;">'
                        
                        # Header: Airline & Price
                        all_cards += f'<div style="padding: 20px 24px; border-bottom: 1px solid #f3f4f6; display: flex; justify-content: space-between; align-items: center; background: #ffffff;">'
                        all_cards += f'<div style="display: flex; align-items: center; gap: 12px;">'
                        all_cards += f'<div style="width: 40px; height: 40px; background: #f3f4f6; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #4b5563;"><svg style="width:20px;height:20px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg></div>'
                        all_cards += f'<div><h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #111827;">{flight["airline"]}</h3><div style="font-size: 12px; color: #6b7280; margin-top: 2px;">Flight {idx}</div></div>'
                        all_cards += f'</div><div style="text-align: right;"><div style="font-size: 20px; font-weight: 800; color: #111827;">{price_amount} <span style="font-size: 14px; font-weight: 500; color: #6b7280;">{price_currency}</span></div></div></div>'
                        
                        # Body: Times & Route
                        all_cards += f'<div style="padding: 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px;">'
                        all_cards += f'<div style="text-align: left; flex: 1;"><div style="font-size: 24px; font-weight: 700; color: #111827; line-height: 1.2;">{flight["departure"]}</div><div style="font-size: 14px; font-weight: 600; color: #9ca3af; margin-top: 4px;">{origin}</div></div>'
                        all_cards += f'<div style="flex: 2; display: flex; flex-direction: column; align-items: center; position: relative; padding: 0 10px;"><div style="font-size: 12px; font-weight: 600; color: #6b7280; margin-bottom: 8px;">{flight["duration"]}</div>'
                        all_cards += f'<div style="width: 100%; height: 2px; background: #e5e7eb; position: relative; border-radius: 2px;"><div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: {stops_color}; padding: 0 8px; font-size: 10px; font-weight: 700; color: white; border-radius: 10px; line-height: 16px; white-space: nowrap;">{stops_text}</div></div></div>'
                        all_cards += f'<div style="text-align: right; flex: 1;"><div style="font-size: 24px; font-weight: 700; color: #111827; line-height: 1.2;">{flight["arrival"]}</div><div style="font-size: 14px; font-weight: 600; color: #9ca3af; margin-top: 4px;">{destination}</div></div></div>'
                        
                        # Footer: Action
                        all_cards += f'<div style="padding: 16px 24px; background: #f9fafb; border-top: 1px solid #f3f4f6; display: flex; align-items: center; justify-content: space-between;">'
                        all_cards += f'<div style="display: flex; gap: 16px; font-size: 12px; font-weight: 500; color: #6b7280;"><span style="display: flex; align-items: center; gap: 4px;">🧳 Included</span><span style="display: flex; align-items: center; gap: 4px;">💺 Economy</span></div>'
                        all_cards += f'<a href="{booking_url}" style="background: #111827; color: white; text-decoration: none; padding: 12px 28px; border-radius: 8px; font-weight: 600; font-size: 14px; transition: all 0.2s; display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">Select Flight →</a></div>'
                        
                        all_cards += '</div>'
                    
                    all_cards += '<div style="text-align:center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;"><p style="color: #9ca3af; font-size: 12px; margin: 0;">Prices include all taxes and fees • 24/7 Support</p></div></div>'
                    return all_cards
                else:
                    return f"<div style='padding: 20px; color: red; background: #fee; border-radius: 8px; font-family: sans-serif;'>{result.get('error')}</div>"
            elif tool_call.function.name == "get_offer":
                args = json.loads(tool_call.function.arguments)
                result = get_offer(offer_id=args.get("offer_id"))
                if result.get("success"):
                    offer = result.get("offer", {})
                    total_amount = result.get("total_amount")
                    total_currency = result.get("total_currency")
                    passenger_ids = result.get("passenger_ids", [])
                    return f"✅ Offer retrieved successfully. Current price: {total_amount} {total_currency}.\n\nIMPORTANT for Agent: You MUST use these Passenger IDs for the booking: {passenger_ids}\n\nPlease collect passenger details (Name, DOB, Gender, Email, Phone + Country Code)."
                elif result.get("expired"):
                    return f"⚠️ {result.get('error')} You can still proceed with booking using the original offer price."
                else:
                    return f"<div style='padding: 20px; color: red; background: #fee; border-radius: 8px; font-family: sans-serif;'>{result.get('error')}</div>"
            elif tool_call.function.name == "create_order":
                args = json.loads(tool_call.function.arguments)
                result = create_order(
                    offer_id=args.get("offer_id"), 
                    passengers=args.get("passengers"), 
                    payment_type=args.get("payment_type", "balance"),
                    total_amount=args.get("total_amount"),
                    total_currency=args.get("total_currency")
                )
                if result.get("success"):
                    booking_ref = result.get("booking_reference")
                    order_id = result.get("order_id")
                    return f"""<div style='padding: 24px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; color: white; font-family: sans-serif;'>
                        <h2 style='margin: 0 0 16px 0; font-size: 24px;'>🎉 Booking Confirmed!</h2>
                        <p style='margin: 8px 0; font-size: 16px;'><strong>Booking Reference:</strong> {booking_ref}</p>
                        <p style='margin: 8px 0; font-size: 16px;'><strong>Order ID:</strong> {order_id}</p>
                        <p style='margin: 16px 0 0 0; font-size: 14px; opacity: 0.9;'>Your flight has been successfully booked. You can use the booking reference to check your reservation on the airline's website.</p>
                    </div>"""
                else:
                    return f"<div style='padding: 20px; color: red; background: #fee; border-radius: 8px; font-family: sans-serif;'>{result.get('error')}</div>"
    
    return response_msg.content or "Hello! I'm your flight booking assistant. Where would you like to fly today?"

agent.deploy(prod=True)
```
