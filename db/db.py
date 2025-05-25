import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Numeric, Date, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func
import psycopg2

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Accident(Base):
    __tablename__ = 'accidents'
    accident_id = Column(Integer, primary_key=True)
    region = Column(String)
    longitude = Column(Numeric)
    latitude = Column(Numeric)
    datetime = Column(Date)
    road_light = Column(Boolean)
    weather = Column(String)
    road_conditions = Column(String)
    severity = Column(Numeric)
    dead_count = Column(Integer)
    injured_count = Column(Integer)
    category = Column(String)
    participants_count = Column(Integer)
    participant_categories = Column(String)
    wet_road = Column(String)
    motorcyclists = Column(Boolean)
    pedestrians = Column(Boolean)
    year = Column(Integer)
    month = Column(Integer)


class Participant(Base):
    __tablename__ = 'participants'
    participant_id = Column(String, primary_key=True)
    accident_id = Column(Integer)
    vehicle_id = Column(String)
    role = Column(String)
    is_male = Column(Boolean)
    years_of_driving_experience = Column(Numeric)
    violations = Column(String)
    health_status = Column(String)
    intoxication = Column(Boolean)
    dl_revocation = Column(Boolean)
    speeding = Column(Boolean)


class Vehicle(Base):
    __tablename__ = 'vehicles'
    vehicle_id = Column(String, primary_key=True)
    accident_id = Column(Integer)
    category = Column(String)
    year_auto = Column(Integer)
    brand = Column(String)
    model = Column(String)
    color = Column(String)


class Region(Base):
    __tablename__ = 'regions'
    region = Column(String, primary_key=True)
    population_region = Column(Integer)
    

__all__ = ['Session', 'Base', 'Region', 'Vehicle','Accident', 'Participant']