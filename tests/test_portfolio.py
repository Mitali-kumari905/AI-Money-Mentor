import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app import app, db
from models import Portfolio, User

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            
            user = User.query.filter_by(email="test@example.com").first()
            if not user:
                user = User(username="testuser", email="test@example.com", password_hash="pbkdf2:sha256:260000$test")
                db.session.add(user)
                db.session.commit()
            user_id = user.id
            
        with client.session_transaction() as sess:
            sess['_user_id'] = str(user_id)
            sess['_fresh'] = True
            
        yield client
        
        with app.app_context():
            db.drop_all()

def test_portfolio_page_render(client):
    res = client.get("/portfolio-page")
    assert res.status_code == 200
    assert b"Portfolio" in res.data
    assert b"Holdings" in res.data or b"holdings" in res.data

def test_add_portfolio_holding(client):
    # Mock stock verification and yfinance info
    mock_stock_price = {
        "symbol": "AAPL",
        "price": 150.0
    }
    
    with patch("app.get_stock_price", return_value=mock_stock_price), \
         patch("app.yf.Ticker") as mock_ticker:
         
        # Mock Ticker.info longName
        mock_instance = MagicMock()
        mock_instance.info = {"longName": "Apple Inc."}
        mock_ticker.return_value = mock_instance
        
        res = client.post("/portfolio/add", json={
            "symbol": "AAPL",
            "quantity": 10,
            "buy_price": 140.0,
            "buy_date": "2026-01-15",
            "notes": "Long-term hold"
        })
        
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["success"] is True
        assert "Successfully added AAPL" in data["message"]
        
        # Verify db item
        with app.app_context():
            item = Portfolio.query.filter_by(symbol="AAPL").first()
            assert item is not None
            assert item.quantity == 10.0
            assert item.buy_price == 140.0
            assert item.buy_date == "2026-01-15"
            assert item.name == "Apple Inc."

def test_add_portfolio_holding_invalid_inputs(client):
    # Missing symbol
    res = client.post("/portfolio/add", json={
        "quantity": 10,
        "buy_price": 140.0,
        "buy_date": "2026-01-15"
    })
    assert res.status_code == 400
    
    # Negative quantity
    res = client.post("/portfolio/add", json={
        "symbol": "AAPL",
        "quantity": -10,
        "buy_price": 140.0,
        "buy_date": "2026-01-15"
    })
    assert res.status_code == 400

    # Invalid date format
    res = client.post("/portfolio/add", json={
        "symbol": "AAPL",
        "quantity": 10,
        "buy_price": 140.0,
        "buy_date": "invalid-date"
    })
    assert res.status_code == 400

def test_delete_portfolio_holding(client):
    # Add a holding directly via db
    with app.app_context():
        user = User.query.first()
        holding = Portfolio(
            user_id=user.id,
            symbol="AAPL",
            name="Apple Inc.",
            quantity=10,
            buy_price=140.0,
            buy_date="2026-01-15"
        )
        db.session.add(holding)
        db.session.commit()
        holding_id = holding.id

    # Delete holding
    res = client.delete(f"/portfolio/delete/{holding_id}")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    
    with app.app_context():
        item = db.session.get(Portfolio, holding_id)
        assert item is None

def test_portfolio_list_calculations(client):
    today = datetime.now()
    
    # Add test holdings to database
    with app.app_context():
        user = User.query.first()
        # Holding A: AAPL
        holding_a = Portfolio(
            user_id=user.id,
            symbol="AAPL",
            name="Apple Inc.",
            quantity=10.0,
            buy_price=100.0,
            buy_date=(today - timedelta(days=200)).strftime("%Y-%m-%d")
        )
        # Holding B: TCS
        holding_b = Portfolio(
            user_id=user.id,
            symbol="TCS",
            name="Tata Consultancy Services",
            quantity=5.0,
            buy_price=200.0,
            buy_date=(today - timedelta(days=50)).strftime("%Y-%m-%d")
        )
        db.session.add(holding_a)
        db.session.add(holding_b)
        db.session.commit()

    # Define mock function for stock prices
    def mock_get_stock_price(symbol):
        if symbol == "AAPL":
            return {"symbol": "AAPL", "price": 110.0}
        if symbol == "TCS":
            return {"symbol": "TCS", "price": 210.0}
        return {"error": "not found"}

    # Define mock dividends lists
    # AAPL dividends
    aapl_divs = [
        {"date": (today - timedelta(days=250)).strftime("%Y-%m-%d"), "amount": 2.0},  # in last 365, before buy date
        {"date": (today - timedelta(days=100)).strftime("%Y-%m-%d"), "amount": 3.0},  # in last 365, after buy date
        {"date": (today - timedelta(days=400)).strftime("%Y-%m-%d"), "amount": 1.5}   # older than 365 days, before buy date
    ]
    # TCS dividends
    tcs_divs = [
        {"date": (today - timedelta(days=20)).strftime("%Y-%m-%d"), "amount": 4.0}    # in last 365, after buy date
    ]

    def mock_get_stock_dividends(symbol):
        if symbol == "AAPL":
            return aapl_divs
        if symbol == "TCS":
            return tcs_divs
        return []

    # Patch the stock fetching and dividend fetching functions
    with patch("app.get_stock_price", side_effect=mock_get_stock_price), \
         patch("app.get_stock_dividends", side_effect=mock_get_stock_dividends):
         
        res = client.get("/portfolio/list")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["success"] is True
        
        # Verify Holdings details
        holdings = {h["symbol"]: h for h in data["holdings"]}
        assert "AAPL" in holdings
        assert "TCS" in holdings
        
        # AAPL detailed assertions
        aapl = holdings["AAPL"]
        assert aapl["quantity"] == 10.0
        assert aapl["buy_price"] == 100.0
        assert aapl["current_price"] == 110.0
        assert aapl["invested_value"] == 1000.0
        assert aapl["current_value"] == 1100.0
        assert aapl["pnl"] == 100.0
        assert aapl["pnl_percent"] == 10.0
        # dividends received = 3.0 (amount) * 10 (quantity) = 30.0 (excluding the 2.0 and 1.5 payouts)
        assert aapl["dividends_received"] == 30.0
        # annual dividend per share = 2.0 + 3.0 = 5.0
        assert aapl["annual_dividend_per_share"] == 5.0
        # YoC = (5.0 / 100.0) * 100 = 5.0%
        assert aapl["yoc"] == 5.0
        
        # TCS detailed assertions
        tcs = holdings["TCS"]
        assert tcs["quantity"] == 5.0
        assert tcs["buy_price"] == 200.0
        assert tcs["current_price"] == 210.0
        assert tcs["invested_value"] == 1000.0
        assert tcs["current_value"] == 1050.0
        assert tcs["pnl"] == 50.0
        assert tcs["pnl_percent"] == 5.0
        # dividends received = 4.0 * 5 = 20.0
        assert tcs["dividends_received"] == 20.0
        # annual dividend per share = 4.0
        assert tcs["annual_dividend_per_share"] == 4.0
        # YoC = (4.0 / 200.0) * 100 = 2.0%
        assert tcs["yoc"] == 2.0
        
        # Summary assertions
        summary = data["summary"]
        assert summary["total_invested"] == 2000.0
        assert summary["total_current"] == 2150.0
        assert summary["total_pnl"] == 150.0
        assert summary["total_pnl_percent"] == 7.5
        assert summary["total_dividends_received"] == 50.0
        # portfolio yoc = ( (5.0 * 10) + (4.0 * 5) ) / 2000.0 * 100 = 70.0 / 2000.0 * 100 = 3.5%
        assert summary["portfolio_yoc"] == 3.5
        
        # Timeline assertions
        timeline = data["timeline"]
        # Payouts that fall in future:
        # AAPL: 250 days ago + 365 = 115 days in future
        # AAPL: 100 days ago + 365 = 265 days in future
        # TCS: 20 days ago + 365 = 345 days in future
        # AAPL 400 days ago + 365 = 35 days in past (filtered out)
        assert len(timeline) == 3
        
        # Verify that it is sorted by date
        dates = [t["date"] for t in timeline]
        assert dates == sorted(dates)
        
        # Verify timeline item details
        # The earliest projected date should be 115 days in future (AAPL)
        expected_date_aapl_1 = (today - timedelta(days=250) + timedelta(days=365)).strftime("%Y-%m-%d")
        assert timeline[0]["symbol"] == "AAPL"
        assert timeline[0]["date"] == expected_date_aapl_1
        assert timeline[0]["amount_per_share"] == 2.0
        assert timeline[0]["amount"] == 20.0  # 2.0 * 10
