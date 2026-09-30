from app.database import Base, engine
from app.models import User, Analysis


def create_tables():
    Base.metadata.create_all(bind=engine)

    print("NetraX database tables created successfully.")


if __name__ == "__main__":
    create_tables()