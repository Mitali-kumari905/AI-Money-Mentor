import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock out external libraries that are not installed before importing app
sys.modules['yfinance'] = MagicMock()
sys.modules['groq'] = MagicMock()
sys.modules['pdfplumber'] = MagicMock()

# Import validation helpers
from utils.validation import ValidationError, validate_string, validate_float, validate_int

# Import the flask app and db
from app import app, db
from models import Expense, Asset, Liability, BudgetLimit, BudgetAlert, User

class TestValidationHelpers(unittest.TestCase):
    """Test cases for the helper validation routines."""

    def test_validate_string(self):
        # Valid string
        self.assertEqual(validate_string("test"), "test")
        self.assertEqual(validate_string("  trimmed  "), "trimmed")
        
        # Missing/None
        with self.assertRaises(ValidationError):
            validate_string(None)
        self.assertIsNone(validate_string(None, allow_none=True))
        
        # Invalid type
        with self.assertRaises(ValidationError):
            validate_string(123)
            
        # Too short
        with self.assertRaises(ValidationError):
            validate_string("", min_length=1)
        with self.assertRaises(ValidationError):
            validate_string("   ", min_length=1)

    def test_validate_float(self):
        # Valid floats
        self.assertEqual(validate_float(10.5), 10.5)
        self.assertEqual(validate_float("10.5"), 10.5)
        self.assertEqual(validate_float(10), 10.0)
        
        # Missing/None
        with self.assertRaises(ValidationError):
            validate_float(None)
        self.assertIsNone(validate_float(None, allow_none=True))
        
        # Invalid types/values
        with self.assertRaises(ValidationError):
            validate_float("not a float")
        with self.assertRaises(ValidationError):
            validate_float(True)  # Booleans rejected
            
        # Min/max boundaries
        self.assertEqual(validate_float(5.0, min_val=5.0), 5.0)
        with self.assertRaises(ValidationError):
            validate_float(4.9, min_val=5.0)
        with self.assertRaises(ValidationError):
            validate_float(10.1, max_val=10.0)

    def test_validate_int(self):
        # Valid ints
        self.assertEqual(validate_int(10), 10)
        self.assertEqual(validate_int("10"), 10)
        self.assertEqual(validate_int(10.0), 10)
        
        # Missing/None
        with self.assertRaises(ValidationError):
            validate_int(None)
        self.assertIsNone(validate_int(None, allow_none=True))
        
        # Invalid types/values
        with self.assertRaises(ValidationError):
            validate_int(5.5)  # Decimal floats rejected
        with self.assertRaises(ValidationError):
            validate_int("not an int")
        with self.assertRaises(ValidationError):
            validate_int(False)  # Booleans rejected
            
        # Min/max boundaries
        self.assertEqual(validate_int(5, min_val=5), 5)
        with self.assertRaises(ValidationError):
            validate_int(4, min_val=5)
        with self.assertRaises(ValidationError):
            validate_int(11, max_val=10)


class TestEndpointValidation(unittest.TestCase):
    """Test cases for Flask endpoints validating request payloads."""

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
        db.create_all()
        
        # Create a test user
        user = User(username="testuser", email="test@example.com", password_hash="pbkdf2:sha256:260000$test")
        db.session.add(user)
        db.session.commit()
        self.user_id = user.id
        
        # Log in user via session
        with self.app.session_transaction() as sess:
            sess['_user_id'] = str(self.user_id)
            sess['_fresh'] = True
            
        # Mock external client in app module
        app_module = sys.modules['app']
        app_module.client = MagicMock()
        mock_ai_res = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Mocked AI recommendation."
        mock_ai_res.choices = [mock_choice]
        app_module.client.chat.completions.create.return_value = mock_ai_res

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_sip_endpoint(self):
        # Valid payload
        response = self.app.post('/sip', json={
            "monthly": 5000,
            "rate": 12.5,
            "years": 10,
            "inflation": 6.0
        })
        self.assertEqual(response.status_code, 200)

        # Missing monthly
        response = self.app.post('/sip', json={
            "rate": 12.5,
            "years": 10
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("monthly", response.get_json()["message"])

        # Negative years
        response = self.app.post('/sip', json={
            "monthly": 5000,
            "rate": 12.5,
            "years": -1
        })
        self.assertEqual(response.status_code, 400)

        # Invalid type (rate as string that cannot be cast)
        response = self.app.post('/sip', json={
            "monthly": 5000,
            "rate": "invalid",
            "years": 10
        })
        self.assertEqual(response.status_code, 400)

    def test_goal_planner_endpoint(self):
        # Valid payload
        response = self.app.post('/goal-planner', json={
            "goal": 1000000,
            "rate": 12.0,
            "years": 5
        })
        self.assertEqual(response.status_code, 200)

        # Missing goal
        response = self.app.post('/goal-planner', json={
            "rate": 12.0,
            "years": 5
        })
        self.assertEqual(response.status_code, 400)

        # Negative rate
        response = self.app.post('/goal-planner', json={
            "goal": 1000000,
            "rate": -5.0,
            "years": 5
        })
        self.assertEqual(response.status_code, 400)

        # Zero years
        response = self.app.post('/goal-planner', json={
            "goal": 1000000,
            "rate": 12.0,
            "years": 0
        })
        self.assertEqual(response.status_code, 400)

    def test_tax_endpoint(self):
        # Valid payload
        response = self.app.post('/tax', json={
            "income": 1200000,
            "deduction_80c": 150000,
            "deduction_80d": 25000
        })
        self.assertEqual(response.status_code, 200)

        # Missing income
        response = self.app.post('/tax', json={
            "deduction_80c": 150000
        })
        self.assertEqual(response.status_code, 400)

        # Negative deduction
        response = self.app.post('/tax', json={
            "income": 1000000,
            "deduction_80c": -100
        })
        self.assertEqual(response.status_code, 400)

    def test_money_score_endpoint(self):
        # Valid payload
        response = self.app.post('/money-score', json={
            "income": 100000,
            "expenses": 40000,
            "savings": 20000,
            "investments": 15000,
            "debt": 10000,
            "emergency": 50000
        })
        self.assertEqual(response.status_code, 200)

        # Missing fields
        response = self.app.post('/money-score', json={
            "income": 100000,
            "expenses": 40000
        })
        self.assertEqual(response.status_code, 400)

        # Negative savings
        response = self.app.post('/money-score', json={
            "income": 100000,
            "expenses": 40000,
            "savings": -10,
            "investments": 15000,
            "debt": 10000,
            "emergency": 50000
        })
        self.assertEqual(response.status_code, 400)

    @patch('app.get_stock_price')
    def test_portfolio_endpoint(self, mock_get_price):
        mock_get_price.return_value = {"symbol": "AAPL", "price": 150.0}
        
        # Valid symbol
        response = self.app.post('/portfolio', json={"stock": "AAPL"})
        self.assertEqual(response.status_code, 200)

        # Invalid/empty stock symbol
        response = self.app.post('/portfolio', json={"stock": ""})
        self.assertEqual(response.status_code, 400)

        # Error from yfinance wrapper
        mock_get_price.return_value = {"error": "Invalid stock symbol or no data found"}
        response = self.app.post('/portfolio', json={"stock": "INVALID"})
        self.assertEqual(response.status_code, 400)

    def test_add_expense_endpoint(self):
        # Valid payload
        response = self.app.post('/add_expense', json={
            "category": "Food",
            "amount": 25.50,
            "date": "2026-06-06"
        })
        self.assertEqual(response.status_code, 200)

        # Missing category
        response = self.app.post('/add_expense', json={
            "amount": 25.50,
            "date": "2026-06-06"
        })
        self.assertEqual(response.status_code, 400)

        # Non-positive amount
        response = self.app.post('/add_expense', json={
            "category": "Food",
            "amount": 0,
            "date": "2026-06-06"
        })
        self.assertEqual(response.status_code, 400)

    def test_add_asset_endpoint(self):
        response = self.app.post('/add-asset', json={
            "name": "Savings Account",
            "amount": 10000
        })
        self.assertEqual(response.status_code, 200)

        # Negative amount
        response = self.app.post('/add-asset', json={
            "name": "Savings Account",
            "amount": -50
        })
        self.assertEqual(response.status_code, 400)

    def test_add_liability_endpoint(self):
        response = self.app.post('/add-liability', json={
            "name": "Home Loan",
            "amount": 500000
        })
        self.assertEqual(response.status_code, 200)

        # Missing name
        response = self.app.post('/add-liability', json={
            "amount": 500000
        })
        self.assertEqual(response.status_code, 400)

    def test_delete_item_endpoint(self):
        # Insert a real asset and liability
        asset = Asset(user_id=self.user_id, name="Gold", amount=50000)
        liability = Liability(user_id=self.user_id, name="Home Loan", amount=500000)
        db.session.add(asset)
        db.session.add(liability)
        db.session.commit()
        
        asset_id = asset.id
        liability_id = liability.id

        # Valid type and id (delete the asset)
        response = self.app.delete('/delete-item', json={
            "type": "asset",
            "id": asset_id
        })
        self.assertEqual(response.status_code, 200)

        # Invalid type
        response = self.app.delete('/delete-item', json={
            "type": "invalid_type",
            "id": liability_id
        })
        self.assertEqual(response.status_code, 400)

        # Invalid id (less than 1)
        response = self.app.delete('/delete-item', json={
            "type": "asset",
            "id": 0
        })
        self.assertEqual(response.status_code, 400)

        # Item not found (404)
        response = self.app.delete('/delete-item', json={
            "type": "asset",
            "id": 99999
        })
        self.assertEqual(response.status_code, 404)

    def test_budget_limits_endpoint(self):
        # Valid payload
        response = self.app.post('/budget/limits', json={
            "category": "Food",
            "limit_amount": 500
        })
        self.assertEqual(response.status_code, 200)

        # Missing limit_amount
        response = self.app.post('/budget/limits', json={
            "category": "Food"
        })
        self.assertEqual(response.status_code, 400)

        # Negative limit_amount
        response = self.app.post('/budget/limits', json={
            "category": "Food",
            "limit_amount": -10
        })
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
