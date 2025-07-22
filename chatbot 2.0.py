# CHATBOT'S PERSONALITY
bot_name = "CryptoBuddy"
intro = f"""
👋 Hey, I'm {bot_name}! 
Your meme-loving 🧠 crypto sidekick here to help you find green 🌱 and growing 📈 coins!
Ask me about sustainability, price trends, or long-term crypto growth.
Type 'exit' anytime to leave the chat.
"""
print(intro)

# PREDEFINED CRYPTO DATA
crypto_db = {
    "Bitcoin": {
        "price_trend": "rising",
        "market_cap": "high",
        "energy_use": "high",
        "sustainability_score": 3/10
    },
    "Ethereum": {
        "price_trend": "stable",
        "market_cap": "high",
        "energy_use": "medium",
        "sustainability_score": 6/10
    },
    "Cardano": {
        "price_trend": "rising",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 8/10
    }
}

# CHATBOT LOGIC


def handle_query(user_query):
    user_query = user_query.lower()

    # Sustainability query
    if "sustainable" in user_query or "eco" in user_query:
        recommend = max(
            crypto_db, key=lambda x: crypto_db[x]["sustainability_score"])
        print(
            f"🌱 Invest in {recommend}! It’s eco-friendly and has long-term potential.")

    # Trending or profitable
    elif "trending" in user_query or "up" in user_query or "profitable" in user_query:
        trending = [coin for coin, data in crypto_db.items()
                    if data["price_trend"] == "rising" and data["market_cap"] == "high"]
        if trending:
            print("📈 These profitable cryptos are trending up:",
                  ", ".join(trending))
        else:
            print("🤷 No high-market-cap coins are trending up right now.")

    # Long-term investment advice
    elif "long-term" in user_query or "growth" in user_query or "invest" in user_query:
        for name, data in crypto_db.items():
            if data["price_trend"] == "rising" and data["sustainability_score"] > 0.7:
                print(
                    f"🚀 {name} is trending up and has a top-tier sustainability score!")
                return
        print("📉 Nothing stands out for long-term growth at the moment.")

    # Help or capabilities
    elif "help" in user_query or "can you do" in user_query:
        print("""
🛠️ I can help with:
- Finding the most sustainable coins 🌱
- Showing which cryptos are trending 📈
- Recommending long-term investments 💼
Just ask me something like “Which coin is most profitable?”
        """)

    # Unknown or unsupported query
    else:
        print("❓ Hmm... I didn't get that. Ask me about 'sustainable', 'trending', or 'growth' coins.")


# TESTING BOT
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Catch you later mate! Remember: Crypto is risky — always DYOR (Do Your Own Research). 🚨")
        break
    handle_query(user_input)
    print("🤖 CryptoBuddy is here to help! Ask me anything about crypto trends, sustainability, or long-term investments.")
'''
SAMPLE PROMPTS
Which crypto is trending up?
Tell me the most profitable coin.
Which coin is most sustainable?
What are your capabilities?
long time investment advice
'''
