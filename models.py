from database import db

class Review(db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    
    text = db.Column(db.Text, nullable=False)
    
    prediction = db.Column(db.String(20))
    
    confidence = db.Column(db.Float)
    
    #bias_score = db.Column(db.Float)

    def __repr__(self):
        return f"<Review {self.id}>"