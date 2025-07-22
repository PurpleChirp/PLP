# 👋Bot Introduction & Personality
bot_name = "CryptoBuddy"
bot_intro = f"""
👋 Hey there Mate! I'm {bot_name} — your friendly AI-powered crypto sidekick.
I’ll help you find neat 🌱 and promising 📈 coins based on trends and sustainability!
Type 'exit' or 'quit' anytime to end our chat.
"""
print(bot_intro)

# 📊Predefined Crypto Data
crypto_db = {
    "Bitcoin": {
        "price_trend": "rising",
        "market_cap": "high",
        "energy_use": "high",
        "sustainability_score": 3
    },
    "Ethereum": {
        "price_trend": "stable",
        "market_cap": "high",
        "energy_use": "medium",
        "sustainability_score": 6
    },
    "Cardano": {
        "price_trend": "rising",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 8
    }
}


def get_most_sustainable():
    return max(crypto_db, key=lambda x: crypto_db[x]["sustainability_score"])


def get_trending():
    return [k for k, v in crypto_db.items() if v["price_trend"] == "rising"]


def get_long_term():
    for name, data in crypto_db.items():
        if data["price_trend"] == "rising" and data["sustainability_score"] >= 7:
            return name
    return None


def handle_query(user_query):
    user_query = user_query.lower()
    response = ""

    if any(word in user_query for word in ("sustainable", "eco")):
        best = get_most_sustainable()
        response = f"🌱 Go for {best}! It's energy-efficient and has a high sustainability score!"
    elif any(word in user_query for word in ("trending", "going up", "rise")):
        trending = get_trending()
        response = "📈 These cryptos are trending up right now: " + \
            ", ".join(trending)
    elif any(word in user_query for word in ("long-term", "growth", "invest")):
        pick = get_long_term()
        if pick:
            response = f"🚀 {pick} is a solid pick for long-term growth and green investing!"
        else:
            response = "🤔 None match both growth and sustainability right now."
    elif "advice" in user_query:
        response = ("📊 Here's how I give advice:\n"
                    "- For profit: rising trend + high market cap\n"
                    "- For sustainability: low energy use + high sustainability score")
    else:
        response = "❓ I'm not sure about that. Try asking about 'sustainability', 'growth', or 'trending'. 💬"

    print(response)


# 🧪Interactive Chat Loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("exit", "quit"):
        print("👋 Stay safe and invest wisely mate! ⚠️ Crypto is risky—always DYOR (Do Your Own Research)!")
        break
    handle_query(user_input)
    print("🤖 CryptoBuddy: How can I help you today?")

'''
SOME EXAMPLE QUERIES:
Which crypto is trending up?
What’s the most sustainable coin?
Which coin should I invest in for long-term growth?
Can you give me some advice?
'''
