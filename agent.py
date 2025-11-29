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

@agent("flightagent", header=header, intro=intro, auth=True)
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

agent.deploy(prod=False)
