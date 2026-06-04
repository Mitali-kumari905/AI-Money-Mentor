from flask import Flask, request, jsonify, render_template
import yfinance as yf
import os
import sys
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import json
import hashlib


# Load environment variables from .env file (if present)
load_dotenv()

# ── Startup validation ───────────────────────────────────────
# Fail fast and clearly if the required API key is missing.
# Copy .env.example → .env and set your GROQ_API_KEY.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY.strip() in ("", "your_groq_api_key_here"):
    print(
        "\n[ERROR] GROQ_API_KEY is not configured.\n"
        "  1. Copy .env.example to .env\n"
        "  2. Set your GROQ_API_KEY in .env\n"
        "  Obtain a free key at: https://console.groq.com/\n",
        file=sys.stderr,
    )
    sys.exit(1)
# ---------------- IMPORT UTILS ----------------
from utils.sip import calculate_sip
from utils.tax import calculate_tax
from utils.pdf_parser import extract_income
from utils.money_score import calculate_money_score
from utils.multi_agent import run_multi_agent
from utils.stock import get_stock_price
from utils.expense_track import calculate_expense, insights
from utils import persistence
from utils.ai_categorizer import AICategorizer

app = Flask(__name__)

# ---------------- INIT DATABASE ----------------
from models import db, Expense, Asset, Liability, Portfolio, PriceAlert

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///money_mentor.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# ---------------- INIT GROQ ----------------
client = Groq(api_key=GROQ_API_KEY)

# Initialize AI Categorizer
ai_categorizer = AICategorizer()

# ── Dev-mode startup message ─────────────────────────────────
if os.getenv("FLASK_ENV", "development") != "production":
    print("[OK] Groq client initialised successfully.")
    print("[OK] AI Expense Categorizer loaded.")

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health_check():
    """Lightweight liveness probe for deployment environments (Docker, Railway, etc.)."""
    return jsonify({"status": "ok", "service": "AI Money Mentor"}), 200


# ---------------- ERROR HANDLERS ----------------
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad Request",
        "message": str(error),
        "status_code": 400
    }), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist.",
        "status_code": 404
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method Not Allowed",
        "message": str(error),
        "status_code": 405
    }), 405


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later.",
        "status_code": 500
    }), 500


# ---------------- 🤖 MULTI-AGENT AI CHAT ----------------
from utils.multi_agent_system import MultiAgentRouter

# Initialize multi-agent router
multi_agent_router = None

def get_router():
    global multi_agent_router
    if multi_agent_router is None:
        multi_agent_router = MultiAgentRouter(client)
    return multi_agent_router

@app.route("/chat", methods=["POST"])
def chat():
    """Multi-agent powered chat with specialized financial advisors"""
    try:
        data = request.json
        msg = data.get("message", "")
        history = data.get("history", [])
        
        if not msg:
            return jsonify({"reply": "Please ask a question about your finances."}), 400
        
        # Use multi-agent system
        router = get_router()
        result = router.process_query(msg, history)
        
        # Format response with agent info (optional, can be hidden)
        reply = result['response']
        
        # Uncomment below to show which agent responded (for debugging)
        # reply = f"**[{result['agent']} - {result['specialization']}]**\n\n{result['response']}"
        
        return jsonify({
            "reply": reply,
            "agent_used": result.get('agent', 'AI Advisor'),
            "specialization": result.get('specialization', 'Finance'),
            "confidence": result.get('confidence', 0.8)
        })
        
    except Exception as e:
        app.logger.error(f"Multi-Agent Chat Error: {str(e)}")
        return jsonify({
            "reply": "I'm here to help with your financial questions. Could you please rephrase your question?"
        }), 200

@app.route("/agent-stats", methods=["GET"])
def agent_stats():
    """Get performance statistics for all agents (admin endpoint)"""
    try:
        router = get_router()
        stats = router.get_performance_stats()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




# ---------------- 💸 SIP ----------------
@app.route("/sip", methods=["POST"])
def sip():
    try:
        data = request.json
        result = calculate_sip(
            float(data["monthly"]),
            float(data["rate"]),
            int(data["years"])
        )
        return jsonify({"future_value": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- 📊 STOCK ----------------
@app.route("/portfolio", methods=["POST"])
def portfolio():
    try:
        stock = request.json["stock"].upper()
        result = get_stock_price(stock)
        
        # AI Sentiment Analysis
        sentiment = "Neutral"
        analysis = "No recent news available to analyze."
        
        if "error" not in result and result.get("news"):
            headlines = [n["title"] for n in result["news"]]
            news_text = "\n".join([f"- {h}" for h in headlines])
            
            prompt = (
                f"Analyze the market sentiment for stock ticker {stock} based on the following recent news headlines:\n"
                f"{news_text}\n\n"
                f"Your output must be in this exact format:\n"
                f"SENTIMENT: [Bullish / Bearish / Neutral]\n"
                f"EXPLANATION: [A concise 2-3 sentence summary explaining why the stock is moving based on the news, or overall outlook if news is mixed.]"
            )
            
            try:
                ai_res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a professional stock market advisor. Be concise and accurate."},
                        {"role": "user", "content": prompt}
                    ]
                )
                ai_output = ai_res.choices[0].message.content.strip()
                
                # Simple parsing of the formatted output
                if "SENTIMENT:" in ai_output:
                    parts = ai_output.split("EXPLANATION:")
                    sent_part = parts[0].replace("SENTIMENT:", "").strip()
                    # Clean sentiment word
                    for option in ["Bullish", "Bearish", "Neutral"]:
                        if option.lower() in sent_part.lower():
                            sentiment = option
                            break
                    if len(parts) > 1:
                        analysis = parts[1].strip()
                    else:
                        analysis = ai_output
                else:
                    analysis = ai_output
            except Exception as ai_err:
                app.logger.error(f"Stock AI Analysis Error: {str(ai_err)}")
                analysis = "AI Sentiment Analysis is currently unavailable."
        
        result["sentiment"] = sentiment
        result["analysis"] = analysis
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# ---------------- 💸 TAX ----------------
@app.route("/tax", methods=["POST"])
def tax():
    try:
        income = float(request.json["income"])
        return jsonify({"tax": calculate_tax(income)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- 📄 PDF ----------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        result = extract_income(file)
        return jsonify({"data": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- 🧠 MULTI AGENT ----------------
@app.route("/agent", methods=["POST"])
def run_agent_route():
    try:
        query = request.json["query"]
        response = run_multi_agent(client, query)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- 💰 MONEY SCORE ----------------
@app.route("/money-score", methods=["POST"])
def money_score():
    try:
        data = request.json

        score = calculate_money_score(
            float(data["income"]),
            float(data["expenses"]),
            float(data["savings"]),
            float(data["investments"]),
            float(data["debt"]),
            float(data["emergency"])
        )

        if score >= 80:
            status = "Excellent 💚"
        elif score >= 60:
            status = "Good 👍"
        elif score >= 40:
            status = "Average ⚠️"
        else:
            status = "Needs Improvement ❌"

        return jsonify({
            "score": score,
            "status": status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- 🤖 AI EXPENSE CATEGORIZATION ----------------

@app.route("/categorize", methods=["POST"])
def categorize_expense():
    """AI endpoint to categorize an expense without saving"""
    try:
        data = request.json
        description = data.get("description", "")
        
        if not description:
            return jsonify({"error": "Description is required"}), 400
        
        result = ai_categorizer.categorize(description)
        
        return jsonify({
            "success": True,
            "description": description,
            "predicted_category": result['category'],
            "confidence": result['confidence'],
            "matched_keywords": result.get('matched_categories', [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/add_expense_ai", methods=["POST"])
def add_expense_ai():
    """Add expense with AI auto-categorization"""
    try:
        data = request.json
        description = data.get("description", "")
        amount = float(data.get("amount", 0))
        date = data.get("date", "")
        
        if not description or not amount:
            return jsonify({"error": "Description and amount are required"}), 400
        
        # Let AI categorize
        ai_result = ai_categorizer.categorize(description)
        
        expense = Expense(
            category=ai_result['category'],
            amount=amount,
            date=date,
            ai_confidence=ai_result['confidence'],
            user_corrected=False,
            original_ai_category=ai_result['category']
        )
        
        db.session.add(expense)
        db.session.commit()
        
        return jsonify({
            "status": "success",
            "expense_id": expense.id,
            "ai_category": ai_result['category'],
            "confidence": ai_result['confidence'],
            "message": f"Expense automatically categorized as {ai_result['category']} with {ai_result['confidence']*100}% confidence"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400


@app.route("/correct_category", methods=["POST"])
def correct_category():
    """User corrects AI category - helps AI learn"""
    try:
        data = request.json
        expense_id = data.get("expense_id")
        correct_category = data.get("correct_category")
        description = data.get("description", "")
        
        expense = Expense.query.get(expense_id)
        if not expense:
            return jsonify({"error": "Expense not found"}), 404
        
        # Store original before correction
        original_category = expense.category
        
        # Update expense
        expense.category = correct_category
        expense.user_corrected = True
        expense.original_ai_category = original_category
        
        # Teach AI
        ai_categorizer.learn_from_correction(description, correct_category)
        
        db.session.commit()
        
        return jsonify({
            "status": "success",
            "message": f"Category corrected from {original_category} to {correct_category}",
            "ai_improved": True
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400


@app.route("/anomaly_detection", methods=["GET"])
def detect_anomalies():
    """Detect unusual spending patterns"""
    try:
        expenses = Expense.query.all()
        if len(expenses) < 5:
            return jsonify({"message": "Need at least 5 expenses for anomaly detection", "anomalies": []})
        
        # Calculate average spending per category
        category_totals = {}
        category_counts = {}
        
        for expense in expenses:
            cat = expense.category
            amount = expense.amount
            
            if cat not in category_totals:
                category_totals[cat] = 0
                category_counts[cat] = 0
            
            category_totals[cat] += amount
            category_counts[cat] += 1
        
        # Calculate averages
        category_avg = {}
        for cat in category_totals:
            category_avg[cat] = category_totals[cat] / category_counts[cat]
        
        # Find anomalies (spending > 2x average)
        anomalies = []
        for expense in expenses:
            avg = category_avg.get(expense.category, expense.amount)
            if expense.amount > avg * 2 and expense.amount > 1000:  # 2x average and >1000
                anomalies.append({
                    "id": expense.id,
                    "description": "AI detected",
                    "category": expense.category,
                    "amount": expense.amount,
                    "date": expense.date,
                    "reason": f"Spent ₹{expense.amount} which is {round(expense.amount/avg, 1)}x higher than your average of ₹{round(avg, 2)}"
                })
        
        return jsonify({"anomalies": anomalies, "total_anomalies": len(anomalies)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/spending_insights", methods=["GET"])
def spending_insights():
    """Get AI-powered spending insights"""
    try:
        expenses = Expense.query.all()
        
        if not expenses:
            return jsonify({"message": "No expenses found", "insights": []})
        
        # Category breakdown
        category_spending = {}
        for expense in expenses:
            cat = expense.category
            if cat not in category_spending:
                category_spending[cat] = 0
            category_spending[cat] += expense.amount
        
        # Find top spending category
        top_category = max(category_spending, key=category_spending.get)
        
        # Calculate monthly average
        from datetime import datetime, timedelta
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        
        recent_total = 0
        for expense in expenses:
            expense_date = datetime.strptime(expense.date, "%Y-%m-%d")
            if expense_date > thirty_days_ago:
                recent_total += expense.amount
        
        insights_list = [
            f"Your top spending category is {top_category} (₹{category_spending[top_category]:,.2f})",
            f"You've spent ₹{recent_total:,.2f} in the last 30 days",
            f"AI confidence in categorizations: {round(sum(e.ai_confidence for e in expenses)/len(expenses)*100)}%",
        ]
        
        # Subscription detection
        subscriptions = [e for e in expenses if e.is_subscription or "subscription" in e.category.lower()]
        if subscriptions:
            insights_list.append(f"Found {len(subscriptions)} potential subscriptions - review them to save money")
        
        return jsonify({"insights": insights_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- EXPENSE TRACKER (Original) ----------------

@app.route("/add_expense", methods=["POST"])
def add_expense():
    try:
        data = request.json
        if not data or "category" not in data or "amount" not in data or "date" not in data:
            return jsonify({"error": "category, amount, and date are required"}), 400

        print("RECEIVED:", data)

        expense = Expense(
            category=str(data["category"]).strip(),
            amount=float(data["amount"]),
            date=str(data["date"]).strip()
        )
        db.session.add(expense)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 400

@app.route("/calculate", methods=["GET"])
def calculate():
    expense_data = [e.to_dict() for e in Expense.query.order_by(Expense.id).all()]
    result = calculate_expense(expense_data)
    result["expenses"] = expense_data
    return jsonify(result)


@app.route("/insights", methods=["GET"])
def expense_insights():
    expense_data = [e.to_dict() for e in Expense.query.order_by(Expense.id).all()]
    result = insights(client, expense_data)
    return jsonify(result)


# ---------------- NET WORTH TRACKER ----------------
@app.route("/net-worth", methods=["GET", "POST"])
def get_net_worth():
    assets = Asset.query.order_by(Asset.id).all()
    liabilities = Liability.query.order_by(Liability.id).all()
    assets_data = [a.to_dict(i) for i, a in enumerate(assets)]
    liabilities_data = [l.to_dict(i) for i, l in enumerate(liabilities)]
    total_assets = sum(item['amount'] for item in assets_data)
    total_liabilities = sum(item['amount'] for item in liabilities_data)
    return jsonify({
        "assets": assets_data,
        "liabilities": liabilities_data,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "net_worth": total_assets - total_liabilities,
    })


@app.route("/add-asset", methods=["POST"])
def add_asset():
    try:
        data = request.json
        if not data or "name" not in data or "amount" not in data:
            return jsonify({"error": "name and amount are required"}), 400
        asset = Asset(name=str(data["name"]).strip(), amount=float(data["amount"]))
        db.session.add(asset)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/add-liability", methods=["POST"])
def add_liability():
    try:
        data = request.json
        if not data or "name" not in data or "amount" not in data:
            return jsonify({"error": "name and amount are required"}), 400
        liability = Liability(name=str(data["name"]).strip(), amount=float(data["amount"]))
        db.session.add(liability)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/delete-item", methods=["POST"])
def delete_item():
    """Delete an asset or liability by its stable id (NOT list index).

    Previously this used list.pop(index) which silently corrupted
    all subsequent indices after the first deletion.
    """
    try:
        data = request.json
        item_type = data.get("type") # 'asset' or 'liability'
        item_id = int(data.get("id")) # positional index from the frontend

        if item_type == 'asset':
            rows = Asset.query.order_by(Asset.id).all()
            db.session.delete(rows[item_id])
        else:
            rows = Liability.query.order_by(Liability.id).all()
            db.session.delete(rows[item_id])

        db.session.commit()
        return jsonify({"status": "success"})
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------- PORTFOLIO TRACKER ----------------

# Simple cache for stock prices (5 minutes)
price_cache = {}
CACHE_DURATION = 300  # 5 seconds for testing, change to 300 for production

def get_cached_price(symbol):
    """Get cached stock price or fetch new one"""
    cache_key = symbol.upper()
    now = datetime.now().timestamp()
    
    if cache_key in price_cache:
        cached_time, cached_price = price_cache[cache_key]
        if now - cached_time < CACHE_DURATION:
            return cached_price
    return None

def set_cached_price(symbol, price):
    """Cache stock price"""
    cache_key = symbol.upper()
    price_cache[cache_key] = (datetime.now().timestamp(), price)

@app.route("/portfolio/add", methods=["POST"])
def add_portfolio_item():
    """Add a stock to portfolio"""
    try:
        data = request.json
        symbol = data.get("symbol", "").upper()
        name = data.get("name", symbol)
        quantity = float(data.get("quantity", 0))
        buy_price = float(data.get("buy_price", 0))
        buy_date = data.get("buy_date", datetime.now().strftime("%Y-%m-%d"))
        investment_type = data.get("investment_type", "stock")
        
        if not symbol or quantity <= 0 or buy_price <= 0:
            return jsonify({"error": "Invalid input"}), 400
        
        # Validate stock symbol
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            info = stock.info
            if not info or 'regularMarketPrice' not in info:
                # Try to get basic info
                hist = stock.history(period="1d")
                if hist.empty:
                    return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
            name = info.get('longName', name)
        except:
            pass
        
        portfolio_item = Portfolio(
            symbol=symbol,
            name=name,
            quantity=quantity,
            buy_price=buy_price,
            buy_date=buy_date,
            investment_type=investment_type
        )
        db.session.add(portfolio_item)
        db.session.commit()
        
        return jsonify({"success": True, "message": f"Added {symbol} to portfolio"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/portfolio/list", methods=["GET"])
def get_portfolio():
    """Get all portfolio items with live prices"""
    try:
        items = Portfolio.query.all()
        portfolio_data = []
        total_invested = 0
        total_current = 0
        
        for item in items:
            # Get live price (with cache)
            cached_price = get_cached_price(item.symbol)
            if cached_price:
                current_price = cached_price
            else:
                try:
                    import yfinance as yf
                    stock = yf.Ticker(item.symbol)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        set_cached_price(item.symbol, current_price)
                    else:
                        current_price = item.buy_price
                except:
                    current_price = item.buy_price
            
            item_data = item.to_dict(current_price)
            portfolio_data.append(item_data)
            total_invested += item_data["invested_value"]
            total_current += item_data["current_value"]
        
        total_pnl = total_current - total_invested
        total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        return jsonify({
            "success": True,
            "holdings": portfolio_data,
            "summary": {
                "total_invested": round(total_invested, 2),
                "total_current": round(total_current, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_percent": round(total_pnl_percent, 2),
                "total_holdings": len(portfolio_data)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/portfolio/delete/<int:item_id>", methods=["DELETE"])
def delete_portfolio_item(item_id):
    """Remove item from portfolio"""
    try:
        item = Portfolio.query.get(item_id)
        if not item:
            return jsonify({"error": "Item not found"}), 404
        
        db.session.delete(item)
        db.session.commit()
        return jsonify({"success": True, "message": "Item removed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/portfolio/alert/add", methods=["POST"])
def add_price_alert():
    """Add price alert for a stock"""
    try:
        data = request.json
        symbol = data.get("symbol", "").upper()
        target_price = float(data.get("target_price", 0))
        condition = data.get("condition", "above")
        
        alert = PriceAlert(
            symbol=symbol,
            target_price=target_price,
            condition=condition
        )
        db.session.add(alert)
        db.session.commit()
        
        return jsonify({"success": True, "message": f"Alert set for {symbol} at ₹{target_price}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/portfolio/alerts", methods=["GET"])
def get_alerts():
    """Get all price alerts"""
    try:
        alerts = PriceAlert.query.all()
        return jsonify({"success": True, "alerts": [a.to_dict() for a in alerts]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/portfolio/check-alerts", methods=["GET"])
def check_price_alerts():
    """Check and trigger price alerts"""
    try:
        alerts = PriceAlert.query.filter_by(is_triggered=False).all()
        triggered = []
        
        for alert in alerts:
            cached_price = get_cached_price(alert.symbol)
            if cached_price:
                current_price = cached_price
            else:
                try:
                    import yfinance as yf
                    stock = yf.Ticker(alert.symbol)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        continue
                except:
                    continue
            
            if alert.condition == "above" and current_price >= alert.target_price:
                alert.is_triggered = True
                triggered.append({"symbol": alert.symbol, "target": alert.target_price, "current": current_price})
            elif alert.condition == "below" and current_price <= alert.target_price:
                alert.is_triggered = True
                triggered.append({"symbol": alert.symbol, "target": alert.target_price, "current": current_price})
        
        db.session.commit()
        return jsonify({"success": True, "triggered": triggered})
    except Exception as e:
        return jsonify({"error": str(e)}), 400




# ---------------- 📅 MONTHLY BUDGET PLANNER ----------------
@app.route("/budget-planner", methods=["GET"])
def budget_planner():
    return render_template("index.html")


@app.route("/budget-planner/analyze", methods=["POST"])
def analyze_budget():
    try:
        data = request.json
        income = float(data.get("income", 0))
        categories = data.get("categories", [])

        if not income or not categories:
            return jsonify({"error": "Income and categories are required"}), 400

        total_budgeted = sum(float(c.get("budgeted", 0)) for c in categories)
        total_spent = sum(float(c.get("spent", 0)) for c in categories)

        budget_lines = "\n".join([
            f"- {c['name']}: Budgeted ₹{c['budgeted']}, Spent ₹{c['spent']} "
            f"({round(float(c['spent']) / float(c['budgeted']) * 100 if float(c.get('budgeted', 0)) > 0 else 0)}% used)"
            for c in categories
        ])

        prompt = (
            f"You are a personal finance advisor. Analyze this monthly budget:\n\n"
            f"Monthly Income: ₹{income}\n"
            f"Total Budgeted: ₹{total_budgeted}\n"
            f"Total Spent: ₹{total_spent}\n"
            f"Unallocated / Remaining: ₹{income - total_spent}\n\n"
            f"Category Breakdown:\n{budget_lines}\n\n"
            f"Give exactly 4 bullet points of concise, actionable advice:\n"
            f"• Overall budget health assessment\n"
            f"• Any category that is over budget or at risk\n"
            f"• One specific saving tip based on the numbers\n"
            f"• Recommended savings target for next month"
        )

        ai_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful personal finance advisor. Be concise and specific with numbers."},
                {"role": "user", "content": prompt}
            ]
        )

        advice = ai_res.choices[0].message.content.strip()

        return jsonify({
            "success": True,
            "advice": advice,
            "summary": {
                "income": income,
                "total_budgeted": round(total_budgeted, 2),
                "total_spent": round(total_spent, 2),
                "remaining": round(income - total_spent, 2),
                "savings_rate": round((income - total_spent) / income * 100, 1) if income > 0 else 0
            }
        })

    except Exception as e:
        app.logger.error(f"Budget Planner Error: {str(e)}")
        return jsonify({"error": str(e)}), 400


# ---------------- RUN ----------------
if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
    app.run(debug=debug_mode)