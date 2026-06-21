from app import app, db
from models import User

app.config["TESTING"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"

with app.app_context():
    db.create_all()
    user = User(username="testuser", email="test@example.com", password_hash="pbkdf2:sha256:260000$test")
    db.session.add(user)
    db.session.commit()
    user_id = user.id
    
    # Query user back
    retrieved_user = db.session.get(User, user_id)
    print(f"Retrieved user directly inside context: {retrieved_user}")

with app.test_client() as client:
    with client.session_transaction() as sess:
        sess['_user_id'] = str(user_id)
        sess['_fresh'] = True
        
    res = client.get("/api/alerts")
    print(f"Response status: {res.status_code}")
    print(f"Response data: {res.data.decode('utf-8')[:200]}")
