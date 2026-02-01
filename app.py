import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium
import folium
import requests
import json
import os
from fpdf import FPDF
import base64
from io import BytesIO
import hashlib
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Climate-TMS Pro | Интеллектуальная логистика",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 2rem;}
    .section-title {font-size: 1.5rem; font-weight: bold; color: #374151; margin: 1.5rem 0 1rem 0;}
    .metric-card {text-align: center; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); background: white;}
    .chat-message {padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #3b82f6;}
    .chat-user {background-color: #dbeafe; border-left-color: #3b82f6;}
    .chat-bot {background-color: #f3f4f6; border-left-color: #9ca3af;}
    .dashboard-card {background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1rem;}
    .profit-positive {color: #10b981; font-weight: bold;}
    .profit-negative {color: #ef4444; font-weight: bold;}
    .auth-form {background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ============================================
# БАЗА ДАННЫХ (для авторизации и истории)
# ============================================

def init_database():
    """Инициализация SQLite базы данных"""
    conn = sqlite3.connect('climate_tms.db')
    cursor = conn.cursor()
    
    # Таблица пользователей
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Таблица расчётов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS calculations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            route TEXT NOT NULL,
            origin_city TEXT,
            dest_city TEXT,
            cargo_type TEXT,
            weight_tons REAL,
            vehicle_type TEXT,
            distance_km REAL,
            base_cost REAL,
            climate_surcharge REAL,
            total_cost REAL,
            total_risk REAL,
            avg_temp REAL,
            road_type TEXT,
            departure_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Таблица сравнений маршрутов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS route_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            comparison_name TEXT,
            routes_data TEXT,
            best_route_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Инициализация БД при запуске
init_database()

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================

@st.cache_data
def load_cities_data():
    """Загрузка расширенной базы городов"""
    if os.path.exists("cities_russia_extended.csv"):
        df = pd.read_csv("cities_russia_extended.csv")
        df = df[df["population"] >= 100000]  # Фильтр по населению
        return df
    else:
        # Минимальный датасет для демонстрации
        return create_minimal_cities_df()

def create_minimal_cities_df():
    """Создание минимального датасета городов"""
    data = {
        "city_name": ["Москва", "Санкт-Петербург", "Екатеринбург", "Новосибирск", "Красноярск",
                     "Иркутск", "Якутск", "Норильск", "Воркута", "Мурманск", "Тюмень", "Сургут"],
        "region": ["Москва", "СПб", "Свердл. обл.", "Новосиб. обл.", "Краснояр. край",
                  "Иркут. обл.", "Якутия", "Краснояр. край", "Коми", "Мурманская обл.",
                  "Тюмен. обл.", "ХМАО"],
        "population": [13088000, 5602000, 1544000, 1633000, 1100000,
                      617000, 336000, 184000, 70000, 279000, 847000, 384000],
        "lat": [55.7558, 59.9343, 56.8519, 55.0084, 56.0184,
               52.2866, 62.0355, 69.3558, 67.4956, 68.9707, 57.1530, 61.2527],
        "lon": [37.6173, 30.3351, 60.6122, 82.9357, 92.8672,
               104.3050, 129.7041, 88.1950, 64.0611, 33.0748, 65.5345, 73.4016],
        "is_remote": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        "logistics_hub": [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    }
    return pd.DataFrame(data)

# ============================================
# КОНФИГУРАЦИЯ СИСТЕМЫ
# ============================================

# Типы грузов (расширенная версия)
CARGO_TYPES = {
    "general": {"name": "Общий груз", "description": "Стандартные грузы", "base_rate_per_km": 50, "weight_limit": 20000,
                "size_categories": {"small": {"name": "Мелкий (до 1 т)", "multiplier": 1.0},
                                   "medium": {"name": "Средний (1-5 т)", "multiplier": 1.2},
                                   "large": {"name": "Крупный (5-20 т)", "multiplier": 1.5}}},
    "refrigerated": {"name": "Температурный груз", "description": "Продукты, фармацевтика", "base_rate_per_km": 80, "weight_limit": 15000,
                     "size_categories": {"small": {"name": "Мелкий (до 1 т)", "multiplier": 1.0},
                                        "medium": {"name": "Средний (1-3 т)", "multiplier": 1.3},
                                        "large": {"name": "Крупный (3-15 т)", "multiplier": 1.7}}},
    "dangerous": {"name": "Опасный груз", "description": "Химикаты, горючие материалы", "base_rate_per_km": 120, "weight_limit": 10000,
                  "size_categories": {"small": {"name": "Мелкий (до 0.5 т)", "multiplier": 1.0},
                                     "medium": {"name": "Средний (0.5-2 т)", "multiplier": 1.4},
                                     "large": {"name": "Крупный (2-10 т)", "multiplier": 2.0}}},
    "oversized": {"name": "Негабаритный груз", "description": "Оборудование, стройматериалы", "base_rate_per_km": 150, "weight_limit": 30000,
                  "size_categories": {"small": {"name": "Мелкий (до 5 т)", "multiplier": 1.0},
                                     "medium": {"name": "Средний (5-15 т)", "multiplier": 1.5},
                                     "large": {"name": "Крупный (15-30 т)", "multiplier": 2.5}}},
    "bulk": {"name": "Насыпной груз", "description": "Щебень, песок, зерно", "base_rate_per_km": 40, "weight_limit": 25000,
             "size_categories": {"small": {"name": "Мелкий (до 5 т)", "multiplier": 1.0},
                                "medium": {"name": "Средний (5-15 т)", "multiplier": 1.1},
                                "large": {"name": "Крупный (15-25 т)", "multiplier": 1.3}}},
    "fragile": {"name": "Хрупкий груз", "description": "Стекло, электроника, мебель", "base_rate_per_km": 90, "weight_limit": 12000,
                "size_categories": {"small": {"name": "Мелкий (до 1 т)", "multiplier": 1.0},
                                   "medium": {"name": "Средний (1-4 т)", "multiplier": 1.4},
                                   "large": {"name": "Крупный (4-12 т)", "multiplier": 1.8}}},
    "live": {"name": "Животные", "description": "Скот, птица (спец. условия)", "base_rate_per_km": 100, "weight_limit": 18000,
             "size_categories": {"small": {"name": "Мелкий (до 2 т)", "multiplier": 1.0},
                                "medium": {"name": "Средний (2-8 т)", "multiplier": 1.3},
                                "large": {"name": "Крупный (8-18 т)", "multiplier": 1.6}}}
}

# Типы транспорта
VEHICLE_TYPES = {
    "truck_small": {"name": "Малотоннажник (1.5-3 т)", "capacity": 3000, "base_cost_multiplier": 0.8, "fuel_consumption": 12},
    "truck_medium": {"name": "Среднетоннажник (5-10 т)", "capacity": 10000, "base_cost_multiplier": 1.0, "fuel_consumption": 20},
    "truck_large": {"name": "Фура (20 т)", "capacity": 20000, "base_cost_multiplier": 1.3, "fuel_consumption": 30},
    "refrigerator": {"name": "Рефрижератор", "capacity": 15000, "base_cost_multiplier": 1.5, "fuel_consumption": 35},
    "tanker": {"name": "Автоцистерна", "capacity": 12000, "base_cost_multiplier": 1.4, "fuel_consumption": 28},
    "dump_truck": {"name": "Самосвал", "capacity": 25000, "base_cost_multiplier": 1.2, "fuel_consumption": 32},
    "flatbed": {"name": "Платформа (негабарит)", "capacity": 30000, "base_cost_multiplier": 1.6, "fuel_consumption": 38},
    "livestock": {"name": "Скотовоз", "capacity": 18000, "base_cost_multiplier": 1.4, "fuel_consumption": 26}
}

# База знаний для чат-бота
CHATBOT_KNOWLEDGE = {
    "климат": [
        "Как влияет климат на стоимость перевозки?",
        "Какие климатические факторы учитываются?",
        "Почему зимой дороже?"
    ],
    "стоимость": [
        "Как рассчитывается стоимость?",
        "Что такое климатическая надбавка?",
        "Какие тарифы на разные типы грузов?"
    ],
    "маршруты": [
        "Какие города доступны?",
        "Как рассчитывается расстояние?",
        "Что такое труднодоступные регионы?"
    ],
    "техника": [
        "Какие типы транспорта доступны?",
        "Как выбрать подходящее ТС?",
        "Какие ограничения по весу?"
    ]
}

CHATBOT_RESPONSES = {
    "как влияет климат": "Климат влияет на стоимость через коэффициент риска. При экстремальных температурах, сильных осадках и ветре увеличивается расход топлива, снижается скорость движения и растёт риск поломок. Наша система автоматически рассчитывает климатическую надбавку от 10% до 100% в зависимости от условий.",
    
    "какие климатические факторы": "Мы учитываем: 1) Температуру (комфортная зона -10..+10°C, вне этой зоны — риск), 2) Осадки (снег/дождь снижают сцепление), 3) Ветер (более 15 м/с — опасные условия), 4) Сезонность (зима на Севере — +40% риск), 5) Тип дороги (зимник, грунтовка).",
    
    "почему зимой дороже": "Зимой в северных регионах стоимость выше из-за: 1) Зимников вместо асфальта (+35% риск), 2) Низких температур (-30°C и ниже), 3) Метелей и ограниченной видимости, 4) Необходимости спецоборудования, 5) Снижения средней скорости на 40-60%.",
    
    "как рассчитывается стоимость": "Формула: Стоимость = Расстояние × Ставка руб/км × Множитель груза × Множитель ТС × (1 + Климат. риск). Ставка зависит от типа груза (40-150 ₽/км), множители — от веса и типа ТС, климатический риск — от погодных условий.",
    
    "что такое климатическая надбавка": "Климатическая надбавка — это дополнительная стоимость, которая компенсирует повышенные риски и расходы при перевозке в сложных климатических условиях. Она рассчитывается автоматически на основе прогноза погоды и может составлять от 10% до 100% от базовой стоимости.",
    
    "какие тарифы": "Тарифы руб/км: Насыпной — 40₽, Общий — 50₽, Температурный — 80₽, Хрупкий — 90₽, Животные — 100₽, Опасный — 120₽, Негабаритный — 150₽. Для крупных грузов применяются повышающие коэффициенты.",
    
    "какие города доступны": "В базе более 100 городов РФ с населением от 100 тыс. человек, включая труднодоступные регионы: Норильск, Якутск, Воркута, Мурманск, Новый Уренгой, Салехард, Певек, Тикси и другие арктические населённые пункты.",
    
    "как рассчитывается расстояние": "Мы используем OSRM API для построения реального автомобильного маршрута с учётом дорог. При недоступности API расстояние рассчитывается по прямой линии (формула гаверсинусов) с коэффициентом 1.3 для компенсации.",
    
    "что такое труднодоступные регионы": "Труднодоступные регионы — это территории с экстремальными климатическими условиями (Крайний Север, Дальний Восток, Сибирь), где: 1) Ограниченная транспортная инфраструктура, 2) Сезонная доступность (зимники), 3) Экстремальные температуры, 4) Высокая стоимость логистики.",
    
    "какие типы транспорта": "Доступны 8 типов: малотоннажник, среднетоннажник, фура, рефрижератор, автоцистерна, самосвал, платформа для негабарита, скотовоз. Выбор зависит от типа и веса груза.",
    
    "как выбрать тс": "Выбор ТС зависит от: 1) Веса груза (не должен превышать грузоподъёмность), 2) Типа груза (температура, опасность, габариты), 3) Маршрута (зимник требует полного привода). Система автоматически предлагает подходящие варианты.",
    
    "ограничения по весу": "Ограничения: Насыпной — до 25т, Общий — до 20т, Негабаритный — до 30т, Температурный — до 15т, Опасный — до 10т, Хрупкий — до 12т, Животные — до 18т. Превышение требует специального разрешения."
}

# ============================================
# ФУНКЦИИ СИСТЕМЫ
# ============================================

# Функции погоды, маршрутизации, расчётов (как в предыдущей версии)
# ... (здесь будут все функции из предыдущего кода)

def get_weather_forecast(lat, lon, use_real_weather=True):
    """Получение прогноза погоды"""
    if use_real_weather:
        try:
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation,windspeed_10m",
                "forecast_days": 2
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            temps = data["hourly"]["temperature_2m"][:24]
            precip = data["hourly"]["precipitation"][:24]
            winds = data["hourly"]["windspeed_10m"][:24]
            
            return {
                "source": "real",
                "avg_temp": round(sum(temps) / len(temps), 1),
                "max_precip": round(max(precip), 1),
                "max_wind": round(max(winds), 1),
                "min_temp": round(min(temps), 1),
                "avg_precip": round(sum(precip) / len(precip), 1),
                "avg_wind": round(sum(winds) / len(winds), 1),
                "hourly_data": {
                    "temps": temps,
                    "precip": precip,
                    "winds": winds,
                    "times": data["hourly"]["time"][:24]
                }
            }
        except Exception as e:
            print(f"Ошибка API погоды: {e}")
    
    return generate_weather(lat)

def generate_weather(lat):
    """Генерация реалистичной погоды"""
    if lat > 68:
        base_temp_range = (-45, -20)
        precip_range = (0, 8)
        wind_range = (5, 30)
    elif lat > 63:
        base_temp_range = (-35, -10)
        precip_range = (0, 10)
        wind_range = (3, 25)
    elif lat > 58:
        base_temp_range = (-25, 0)
        precip_range = (0, 12)
        wind_range = (2, 20)
    else:
        base_temp_range = (-15, 10)
        precip_range = (0, 15)
        wind_range = (1, 15)
    
    temps = [np.random.uniform(*base_temp_range) + np.random.normal(0, 3) for _ in range(24)]
    precip = [np.random.exponential(2.0) if np.random.random() < 0.4 else 0.0 for _ in range(24)]
    winds = [np.random.uniform(*wind_range) for _ in range(24)]
    
    return {
        "source": "generated",
        "avg_temp": round(sum(temps) / len(temps), 1),
        "max_precip": round(max(precip), 1),
        "max_wind": round(max(winds), 1),
        "min_temp": round(min(temps), 1),
        "avg_precip": round(sum(precip) / len(precip), 1),
        "avg_wind": round(sum(winds) / len(winds), 1),
        "hourly_data": {
            "temps": temps,
            "precip": precip,
            "winds": winds,
            "times": [(datetime.now() + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(24)]
        }
    }

def get_route_distance_osrm(origin_lat, origin_lon, dest_lat, dest_lon):
    """Получение маршрута через OSRM API"""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=full&geometries=geojson"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data["code"] == "Ok":
            distance_km = data["routes"][0]["distance"] / 1000
            duration_hours = data["routes"][0]["duration"] / 3600
            
            route_geometry = data["routes"][0]["geometry"]["coordinates"]
            route_coords = [[coord[1], coord[0]] for coord in route_geometry]
            
            return {
                "distance_km": round(distance_km, 1),
                "duration_hours": round(duration_hours, 1),
                "route_coords": route_coords,
                "source": "osrm"
            }
        else:
            return None
    except Exception as e:
        print(f"Ошибка OSRM: {e}")
        return None

def calculate_distance_haversine(lat1, lon1, lat2, lon2):
    """Расчёт расстояния по прямой"""
    R = 6371
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def calculate_climate_risk(avg_temp, max_precip, max_wind, road_type, month, lat):
    """Расчёт коэффициента климатического риска"""
    road_risk_map = {"Асфальт": 0.0, "Грунтовая": 0.15, "Зимник": 0.35}
    road_risk = road_risk_map[road_type]
    
    if lat > 60:
        if month in [11, 12, 1, 2, 3]:
            seasonal_risk = 0.40
        elif month in [4, 5, 9, 10]:
            seasonal_risk = 0.20
        else:
            seasonal_risk = 0.0
    else:
        seasonal_risk = 0.0
    
    temp_risk = abs(avg_temp) / 40 if abs(avg_temp) > 10 else 0
    precip_risk = min(max_precip / 10, 0.5)
    wind_risk = max(0, (max_wind - 10) / 20) if max_wind > 10 else 0
    
    weather_risk = 0.5 * temp_risk + 0.3 * precip_risk + 0.2 * wind_risk
    total_risk = min(road_risk + seasonal_risk + weather_risk, 1.0)
    
    return {
        "total_risk": round(total_risk, 3),
        "road_risk": round(road_risk, 3),
        "seasonal_risk": round(seasonal_risk, 3),
        "weather_risk": round(weather_risk, 3),
        "temp_risk": round(temp_risk, 3),
        "precip_risk": round(precip_risk, 3),
        "wind_risk": round(wind_risk, 3)
    }

def determine_road_type(lat):
    """Определение типа дороги по широте"""
    if lat > 65:
        return "Зимник"
    elif lat > 60:
        return "Грунтовая"
    else:
        return "Асфальт"

def calculate_profit(revenue, total_cost, additional_costs=0):
    """Расчёт прибыли"""
    profit = revenue - total_cost - additional_costs
    profit_margin = (profit / revenue * 100) if revenue > 0 else 0
    return {
        "profit": round(profit, 2),
        "profit_margin": round(profit_margin, 1),
        "is_profitable": profit > 0
    }

# ============================================
# ML МОДЕЛЬ ДЛЯ ПРОГНОЗА СПРОСА
# ============================================

def train_demand_forecast_model():
    """Обучение ML-модели для прогноза спроса"""
    # Генерация синтетических данных для обучения
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    for _ in range(n_samples):
        month = np.random.randint(1, 13)
        day_of_week = np.random.randint(0, 7)
        temperature = np.random.uniform(-40, 30)
        is_remote = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Базовый спрос + сезонность + погода + удалённость
        base_demand = 100
        seasonal_factor = 1.5 if month in [11, 12, 1] else 0.8 if month in [6, 7, 8] else 1.0
        weather_factor = 0.7 if temperature < -25 or temperature > 25 else 1.0
        remote_factor = 0.5 if is_remote else 1.0
        
        demand = base_demand * seasonal_factor * weather_factor * remote_factor
        demand += np.random.normal(0, 20)  # Шум
        
        data.append({
            "month": month,
            "day_of_week": day_of_week,
            "temperature": temperature,
            "is_remote": is_remote,
            "demand": max(0, demand)
        })
    
    df = pd.DataFrame(data)
    
    # Обучение модели
    X = df[["month", "day_of_week", "temperature", "is_remote"]]
    y = df["demand"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Сохранение модели и скалера
    with open('demand_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    return model, scaler

def predict_demand(month, day_of_week, temperature, is_remote):
    """Прогноз спроса на перевозки"""
    try:
        with open('demand_model.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            scaler = data['scaler']
        
        X = np.array([[month, day_of_week, temperature, is_remote]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        return round(prediction, 0)
    except:
        # Если модель не обучена — используем простую эвристику
        base_demand = 100
        seasonal_factor = 1.5 if month in [11, 12, 1] else 0.8 if month in [6, 7, 8] else 1.0
        weather_factor = 0.7 if temperature < -25 or temperature > 25 else 1.0
        remote_factor = 0.5 if is_remote else 1.0
        
        return round(base_demand * seasonal_factor * weather_factor * remote_factor, 0)

# ============================================
# АВТОРИЗАЦИЯ
# ============================================

def hash_password(password):
    """Хеширование пароля"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Регистрация нового пользователя"""
    conn = sqlite3.connect('climate_tms.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
            (username, hash_password(password), email)
        )
        conn.commit()
        return True, "Регистрация успешна!"
    except sqlite3.IntegrityError:
        return False, "Пользователь с таким именем уже существует!"
    finally:
        conn.close()

def authenticate_user(username, password):
    """Аутентификация пользователя"""
    conn = sqlite3.connect('climate_tms.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
        (username, hash_password(password))
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return True, {"user_id": result[0], "username": result[1]}
    else:
        return False, None

def save_calculation(user_id, calculation_data):
    """Сохранение расчёта в БД"""
    conn = sqlite3.connect('climate_tms.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO calculations 
        (user_id, route, origin_city, dest_city, cargo_type, weight_tons, vehicle_type,
         distance_km, base_cost, climate_surcharge, total_cost, total_risk, avg_temp,
         road_type, departure_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        calculation_data["route"],
        calculation_data["origin_city"],
        calculation_data["dest_city"],
        calculation_data["cargo_type"],
        calculation_data["weight_tons"],
        calculation_data["vehicle_type"],
        calculation_data["distance_km"],
        calculation_data["base_cost"],
        calculation_data["climate_surcharge"],
        calculation_data["total_cost"],
        calculation_data["total_risk"],
        calculation_data["avg_temp"],
        calculation_data["road_type"],
        calculation_data["departure_date"]
    ))
    
    conn.commit()
    conn.close()

def get_user_calculations(user_id):
    """Получение истории расчётов пользователя"""
    conn = sqlite3.connect('climate_tms.db')
    df = pd.read_sql_query(
        "SELECT * FROM calculations WHERE user_id = ? ORDER BY created_at DESC",
        conn,
        params=(user_id,)
    )
    conn.close()
    return df

# ============================================
# ГЕНЕРАЦИЯ PDF
# ============================================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Climate-TMS Pro | Отчёт о расчёте', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(calculation_data):
    """Генерация PDF отчёта"""
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Основная информация
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Расчёт маршрута', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Маршрут: {calculation_data["route"]}', 0, 1)
    pdf.cell(0, 8, f'Дата отправления: {calculation_data["departure_date"]}', 0, 1)
    pdf.cell(0, 8, f'Тип груза: {calculation_data["cargo_type"]}', 0, 1)
    pdf.cell(0, 8, f'Вес груза: {calculation_data["weight_tons"]} т', 0, 1)
    pdf.cell(0, 8, f'Тип ТС: {calculation_data["vehicle_type"]}', 0, 1)
    pdf.ln(5)
    
    # Стоимость
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Расчёт стоимости', 0, 1)
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Расстояние: {calculation_data["distance_km"]:.0f} км', 0, 1)
    pdf.cell(0, 8, f'Тип дороги: {calculation_data["road_type"]}', 0, 1)
    pdf.cell(0, 8, f'Базовая стоимость: {calculation_data["base_cost"]:,.0f} ₽', 0, 1)
    pdf.cell(0, 8, f'Климатическая надбавка: {calculation_data["climate_surcharge"]:,.0f} ₽ ({calculation_data["total_risk"]*100:.1f}%)', 0, 1)
    pdf.cell(0, 8, f'Итоговая стоимость: {calculation_data["total_cost"]:,.0f} ₽', 0, 1)
    pdf.ln(5)
    
    # Погодные условия
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Погодные условия', 0, 1)
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Средняя температура: {calculation_data["avg_temp"]}°C', 0, 1)
    pdf.cell(0, 8, f'Уровень риска: {calculation_data["risk_level"].upper()}', 0, 1)
    pdf.ln(5)
    
    # Рекомендации
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Рекомендации', 0, 1)
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 11)
    for rec in calculation_data.get("recommendations", []):
        pdf.multi_cell(0, 8, f'• {rec}', 0, 1)
    
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, 'Отчёт сформирован системой Climate-TMS Pro', 0, 1)
    pdf.cell(0, 8, f'Дата формирования: {datetime.now().strftime("%d.%m.%Y %H:%M")}', 0, 1)
    
    # Сохранение в буфер
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin-1')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)
    
    return pdf_buffer

def download_pdf_button(pdf_buffer, filename="calculation_report.pdf"):
    """Кнопка скачивания PDF"""
    pdf_bytes = pdf_buffer.getvalue()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="application/pdf;base64,{b64}" download="{filename}" style="display:inline-block;background:#3b82f6;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">📥 Скачать отчёт (PDF)</a>'
    return href

# ============================================
# ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ============================================

# Инициализация состояния
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "calculator"
if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []
if 'comparison_routes' not in st.session_state:
    st.session_state.comparison_routes = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'ml_model_trained' not in st.session_state:
    st.session_state.ml_model_trained = False

# Загрузка данных о городах
cities_df = load_cities_data()

# Проверка и обучение ML-модели
if not st.session_state.ml_model_trained:
    with st.spinner("Обучение ML-модели прогноза спроса..."):
        train_demand_forecast_model()
        st.session_state.ml_model_trained = True

# ============================================
# СИСТЕМА НАВИГАЦИИ
# ============================================

# Боковая панель навигации
with st.sidebar:
    st.title("🧭 Навигация")
    
    if st.session_state.logged_in:
        st.success(f"👤 {st.session_state.user_data['username']}")
        
        page = st.radio(
            "Выберите раздел:",
            ["Калькулятор маршрутов", "История расчётов", "Сравнение маршрутов", 
             "Прогноз спроса", "Dashboard", "Чат-бот", "Калькулятор прибыли"],
            index=["calculator", "history", "comparison", "demand", "dashboard", "chat", "profit"].index(st.session_state.current_page)
        )
        
        if st.button("🚪 Выйти", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.session_state.current_page = "calculator"
            st.rerun()
    else:
        page = "calculator"
        st.info("Для доступа ко всем функциям войдите или зарегистрируйтесь")
        
        if st.button("🔐 Войти / Регистрация", use_container_width=True):
            st.session_state.current_page = "auth"
            st.rerun()
    
    # Карта сайта
    st.divider()
    st.caption("🗺️ Карта функций")
    st.caption("Уровень 1: ✅ История, ✅ PDF, ✅ Сравнение, ✅ Прибыль")
    st.caption("Уровень 2: 📈 Спрос, 🗣️ Чат-бот, 📱 Мобильный, 📧 Уведомления")
    st.caption("Уровень 3: 🤖 ML, 📊 Dashboard, 🔐 Авторизация, 💳 Платежи")

# Определение текущей страницы
page_map = {
    "Калькулятор маршрутов": "calculator",
    "История расчётов": "history",
    "Сравнение маршрутов": "comparison",
    "Прогноз спроса": "demand",
    "Dashboard": "dashboard",
    "Чат-бот": "chat",
    "Калькулятор прибыли": "profit"
}
st.session_state.current_page = page_map.get(page, "calculator")

# ============================================
# СТРАНИЦА АВТОРИЗАЦИИ
# ============================================

if st.session_state.current_page == "auth" and not st.session_state.logged_in:
    st.markdown('<p class="main-header">🔐 Авторизация</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Вход", "Регистрация"])
    
    with tab1:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        username = st.text_input("Имя пользователя", key="login_username")
        password = st.text_input("Пароль", type="password", key="login_password")
        
        if st.button("Войти", type="primary", use_container_width=True):
            success, user_data = authenticate_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_data = user_data
                st.session_state.current_page = "calculator"
                st.success("✅ Вход выполнен успешно!")
                st.rerun()
            else:
                st.error("❌ Неверное имя пользователя или пароль")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        new_username = st.text_input("Имя пользователя", key="reg_username")
        new_password = st.text_input("Пароль", type="password", key="reg_password")
        new_password_confirm = st.text_input("Подтвердите пароль", type="password", key="reg_password_confirm")
        new_email = st.text_input("Email (опционально)", key="reg_email")
        
        if st.button("Зарегистрироваться", type="primary", use_container_width=True):
            if new_password != new_password_confirm:
                st.error("❌ Пароли не совпадают")
            elif len(new_password) < 6:
                st.error("❌ Пароль должен быть не менее 6 символов")
            else:
                success, message = register_user(new_username, new_password, new_email)
                if success:
                    st.success(f"✅ {message} Теперь войдите в систему.")
                else:
                    st.error(f"❌ {message}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# ============================================
# ГЛАВНАЯ СТРАНИЦА: КАЛЬКУЛЯТОР МАРШРУТОВ
# ============================================

if st.session_state.current_page == "calculator":
    st.markdown('<p class="main-header">🚚 Climate-TMS Pro<br><small>Интеллектуальная система расчёта логистических маршрутов</small></p>', unsafe_allow_html=True)
    
    # Быстрый выбор для незарегистрированных
    if not st.session_state.logged_in:
        st.warning("ℹ️ Вы используете демо-режим. Для сохранения расчётов и доступа ко всем функциям войдите в систему.")
    
    # Параметры расчёта
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Маршрут")
        
        origin_city = st.selectbox(
            "Откуда:",
            options=cities_df["city_name"].tolist(),
            index=cities_df[cities_df["city_name"] == "Москва"].index[0] if "Москва" in cities_df["city_name"].values else 0
        )
        
        dest_city = st.selectbox(
            "Куда:",
            options=cities_df["city_name"].tolist(),
            index=cities_df[cities_df["city_name"] == "Екатеринбург"].index[0] if "Екатеринбург" in cities_df["city_name"].values else 1
        )
        
        origin_data = cities_df[cities_df["city_name"] == origin_city].iloc[0]
        dest_data = cities_df[cities_df["city_name"] == dest_city].iloc[0]
        
        # Индикация труднодоступности
        if origin_data.get("is_remote", 0) == 1 or dest_data.get("is_remote", 0) == 1:
            st.warning("⚠️ Маршрут включает труднодоступный регион. Ожидайте повышенные риски и стоимость.")
    
    with col2:
        st.subheader("📦 Параметры груза")
        
        cargo_type_key = st.selectbox(
            "Тип груза:",
            options=list(CARGO_TYPES.keys()),
            format_func=lambda x: CARGO_TYPES[x]["name"]
        )
        
        cargo_type = CARGO_TYPES[cargo_type_key]
        
        size_category = st.selectbox(
            "Категория груза:",
            options=list(cargo_type["size_categories"].keys()),
            format_func=lambda x: cargo_type["size_categories"][x]["name"]
        )
        
        weight_tons = st.slider(
            f"Вес груза (до {cargo_type['weight_limit']/1000} т):",
            min_value=0.1,
            max_value=cargo_type['weight_limit']/1000,
            value=min(5.0, cargo_type['weight_limit']/1000),
            step=0.5
        )
        
        st.subheader("🚛 Тип транспорта")
        
        vehicle_type_key = st.selectbox(
            "Тип ТС:",
            options=list(VEHICLE_TYPES.keys()),
            format_func=lambda x: VEHICLE_TYPES[x]["name"]
        )
        
        vehicle_type = VEHICLE_TYPES[vehicle_type_key]
    
    # Дополнительные параметры
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        departure_date = st.date_input("📅 Дата отправления", value=datetime.now())
        month = departure_date.month
        
    with col4:
        weather_mode = st.radio(
            "🌦️ Источник погоды:",
            ["Авто (реальная + резерв)", "Только сгенерированная"],
            index=0,
            horizontal=True
        )
    
    # Кнопки расчёта
    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        calculate_button = st.button("🚀 Рассчитать маршрут", type="primary", use_container_width=True, icon="🚀")
    
    with col_btn2:
        recalculate_button = st.button("🔄 Пересчитать", use_container_width=True, icon="🔄")
    
    # РАСЧЁТ МАРШРУТА
    if calculate_button or recalculate_button:
        with st.spinner("Расчёт маршрута..."):
            # Расчёт маршрута
            route_data = get_route_distance_osrm(
                origin_data["lat"], origin_data["lon"],
                dest_data["lat"], dest_data["lon"]
            )
            
            if route_
                distance_km = route_data["distance_km"]
                route_coords = route_data["route_coords"]
                use_osrm = True
            else:
                distance_km = calculate_distance_haversine(
                    origin_data["lat"], origin_data["lon"],
                    dest_data["lat"], dest_data["lon"]
                )
                route_coords = [
                    [origin_data["lat"], origin_data["lon"]],
                    [dest_data["lat"], dest_data["lon"]]
                ]
                use_osrm = False
            
            # Погода
            use_real = (weather_mode == "Авто (реальная + резерв)")
            weather_data = get_weather_forecast(
                (origin_data["lat"] + dest_data["lat"]) / 2,
                (origin_data["lon"] + dest_data["lon"]) / 2,
                use_real_weather=use_real
            )
            
            # Дорога и риски
            mid_lat = (origin_data["lat"] + dest_data["lat"]) / 2
            road_type = determine_road_type(mid_lat)
            risk_data = calculate_climate_risk(
                weather_data["avg_temp"],
                weather_data["max_precip"],
                weather_data["max_wind"],
                road_type,
                month,
                mid_lat
            )
            
            # Стоимость
            base_rate_per_km = cargo_type["base_rate_per_km"]
            size_multiplier = cargo_type["size_categories"][size_category]["multiplier"]
            vehicle_multiplier = vehicle_type["base_cost_multiplier"]
            
            base_cost = distance_km * base_rate_per_km * size_multiplier * vehicle_multiplier
            total_risk = risk_data["total_risk"]
            climate_surcharge = base_cost * total_risk
            total_cost = base_cost + climate_surcharge
            
            # Время в пути
            base_speed = {"Асфальт": 60, "Грунтовая": 40, "Зимник": 25}[road_type]
            weather_penalty = 1 - (risk_data["weather_risk"] * 0.6)
            seasonal_penalty = 1 - (risk_data["seasonal_risk"] * 0.5)
            avg_speed = base_speed * weather_penalty * seasonal_penalty
            estimated_hours = distance_km / avg_speed if avg_speed > 0 else float('inf')
            
            # Уровень риска
            if total_risk >= 0.6:
                risk_level = "critical"
                risk_emoji = "🔴"
            elif total_risk >= 0.3:
                risk_level = "high"
                risk_emoji = "🟡"
            elif total_risk >= 0.1:
                risk_level = "medium"
                risk_emoji = "🟠"
            else:
                risk_level = "low"
                risk_emoji = "🟢"
            
            # Сохранение расчёта
            calculation_data = {
                "route": f"{origin_city} → {dest_city}",
                "origin_city": origin_city,
                "dest_city": dest_city,
                "cargo_type": cargo_type["name"],
                "weight_tons": weight_tons,
                "vehicle_type": vehicle_type["name"],
                "distance_km": distance_km,
                "base_cost": base_cost,
                "climate_surcharge": climate_surcharge,
                "total_cost": total_cost,
                "total_risk": total_risk,
                "avg_temp": weather_data["avg_temp"],
                "road_type": road_type,
                "departure_date": departure_date.strftime("%d.%m.%Y"),
                "risk_level": risk_level,
                "estimated_hours": estimated_hours,
                "avg_speed": avg_speed,
                "weather_source": weather_data["source"]
            }
            
            st.session_state.calculation_history.append(calculation_data)
            
            # Сохранение в БД для авторизованных
            if st.session_state.logged_in:
                save_calculation(st.session_state.user_data["user_id"], calculation_data)
            
            # Отображение результатов
            st.success("✅ Расчёт выполнен успешно!")
            
            st.subheader("💰 Результаты расчёта")
            
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            
            with col_res1:
                st.metric("Базовая стоимость", f"{base_cost:,.0f} ₽")
            with col_res2:
                st.metric("Климат. надбавка", f"{climate_surcharge:,.0f} ₽", f"+{total_risk*100:.0f}%")
            with col_res3:
                st.metric("Итоговая стоимость", f"{total_cost:,.0f} ₽")
            with col_res4:
                st.metric("Уровень риска", f"{risk_emoji} {risk_level.upper()}", f"{total_risk*100:.1f}%")
            
            # Карта
            st.divider()
            st.subheader("🗺️ Карта маршрута")
            
            map_center_lat = (origin_data["lat"] + dest_data["lat"]) / 2
            map_center_lon = (origin_data["lon"] + dest_data["lon"]) / 2
            
            m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=5)
            
            folium.Marker(
                [origin_data["lat"], origin_data["lon"]],
                popup=f"Отправление: {origin_city}",
                icon=folium.Icon(color="green", icon="play")
            ).add_to(m)
            
            folium.Marker(
                [dest_data["lat"], dest_data["lon"]],
                popup=f"Назначение: {dest_city}",
                icon=folium.Icon(color="red", icon="flag")
            ).add_to(m)
            
            colors = {"critical": "red", "high": "orange", "medium": "yellow", "low": "green"}
            folium.PolyLine(
                route_coords,
                color=colors[risk_level],
                weight=5,
                opacity=0.9,
                popup=f"Расстояние: {distance_km:.0f} км\nРиск: {risk_level.upper()}"
            ).add_to(m)
            
            st_folium(m, width="100%", height=400)
            
            # Экспорт в PDF
            st.divider()
            st.subheader("📄 Экспорт отчёта")
            
            pdf_buffer = generate_pdf_report(calculation_data)
            st.markdown(download_pdf_button(pdf_buffer), unsafe_allow_html=True)
            
            # Добавить в сравнение
            if st.button("➕ Добавить в сравнение маршрутов", use_container_width=True):
                if len(st.session_state.comparison_routes) >= 3:
                    st.warning("⚠️ Максимум 3 маршрута для сравнения")
                else:
                    st.session_state.comparison_routes.append(calculation_data)
                    st.success("✅ Маршрут добавлен в сравнение")
            
            # Сохранить в избранное (для авторизованных)
            if st.session_state.logged_in:
                if st.button("⭐ Сохранить в избранное", use_container_width=True):
                    st.success("✅ Маршрут сохранён в вашу историю")

# ============================================
# СТРАНИЦА ИСТОРИИ РАСЧЁТОВ
# ============================================

elif st.session_state.current_page == "history":
    st.markdown('<p class="main-header">📋 История расчётов</p>', unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("ℹ️ Авторизуйтесь для просмотра истории расчётов")
        st.stop()
    
    # Загрузка истории из БД
    history_df = get_user_calculations(st.session_state.user_data["user_id"])
    
    if len(history_df) == 0:
        st.info("📭 История расчётов пуста. Выполните первый расчёт!")
    else:
        st.metric("Всего расчётов", len(history_df))
        
        # Фильтры
        col_filt1, col_filt2, col_filt3 = st.columns(3)
        
        with col_filt1:
            filter_cargo = st.multiselect(
                "Тип груза",
                options=history_df["cargo_type"].unique(),
                default=history_df["cargo_type"].unique()
            )
        
        with col_filt2:
            filter_risk = st.multiselect(
                "Уровень риска",
                options=["low", "medium", "high", "critical"],
                default=["low", "medium", "high", "critical"]
            )
        
        with col_filt3:
            date_range = st.date_input(
                "Период",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        # Применение фильтров
        filtered_df = history_df[
            (history_df["cargo_type"].isin(filter_cargo)) &
            (history_df["total_risk"].apply(lambda x: "low" if x < 0.1 else "medium" if x < 0.3 else "high" if x < 0.6 else "critical").isin(filter_risk))
        ]
        
        # Отображение таблицы
        display_df = filtered_df[[
            "created_at", "route", "cargo_type", "weight_tons", 
            "distance_km", "total_cost", "total_risk", "road_type"
        ]].rename(columns={
            "created_at": "Дата",
            "route": "Маршрут",
            "cargo_type": "Груз",
            "weight_tons": "Вес (т)",
            "distance_km": "Расстояние (км)",
            "total_cost": "Стоимость (₽)",
            "total_risk": "Риск",
            "road_type": "Дорога"
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # Статистика
        st.divider()
        st.subheader("📊 Статистика")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Средняя стоимость", f"{filtered_df['total_cost'].mean():,.0f} ₽")
        with col_stat2:
            st.metric("Средний риск", f"{filtered_df['total_risk'].mean()*100:.1f}%")
        with col_stat3:
            st.metric("Самый дорогой маршрут", f"{filtered_df['total_cost'].max():,.0f} ₽")
        with col_stat4:
            st.metric("Самый рискованный", f"{filtered_df['total_risk'].max()*100:.1f}%")
        
        # Графики
        st.divider()
        st.subheader("📈 Графики")
        
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            fig_costs = px.bar(
                filtered_df.groupby('cargo_type')['total_cost'].mean().reset_index(),
                x='cargo_type',
                y='total_cost',
                title="Средняя стоимость по типам грузов",
                labels={"cargo_type": "Тип груза", "total_cost": "Стоимость (₽)"}
            )
            st.plotly_chart(fig_costs, use_container_width=True)
        
        with col_graph2:
            fig_risks = px.pie(
                filtered_df,
                names=filtered_df["total_risk"].apply(lambda x: "low" if x < 0.1 else "medium" if x < 0.3 else "high" if x < 0.6 else "critical"),
                title="Распределение по уровню риска"
            )
            st.plotly_chart(fig_risks, use_container_width=True)

# ============================================
# СТРАНИЦА СРАВНЕНИЯ МАРШРУТОВ
# ============================================

elif st.session_state.current_page == "comparison":
    st.markdown('<p class="main-header">🆚 Сравнение маршрутов</p>', unsafe_allow_html=True)
    
    if len(st.session_state.comparison_routes) == 0:
        st.info("📭 Нет маршрутов для сравнения. Добавьте маршруты из калькулятора!")
    else:
        st.metric("Маршрутов для сравнения", len(st.session_state.comparison_routes))
        
        if st.button("🗑️ Очистить сравнение", use_container_width=True):
            st.session_state.comparison_routes = []
            st.rerun()
        
        # Таблица сравнения
        comparison_df = pd.DataFrame(st.session_state.comparison_routes)
        
        display_comparison = comparison_df[[
            "route", "cargo_type", "distance_km", "base_cost", 
            "climate_surcharge", "total_cost", "total_risk", "estimated_hours"
        ]].rename(columns={
            "route": "Маршрут",
            "cargo_type": "Груз",
            "distance_km": "Расстояние (км)",
            "base_cost": "Базовая (₽)",
            "climate_surcharge": "Надбавка (₽)",
            "total_cost": "Итого (₽)",
            "total_risk": "Риск",
            "estimated_hours": "Время (ч)"
        })
        
        st.dataframe(display_comparison, use_container_width=True)
        
        # Рекомендация лучшего маршрута
        best_route_idx = comparison_df["total_cost"].idxmin()
        best_route = comparison_df.iloc[best_route_idx]
        
        st.success(f"🏆 **Рекомендуемый маршрут:** {best_route['route']}")
        st.info(f"💰 Экономия: {comparison_df['total_cost'].max() - best_route['total_cost']:,.0f} ₽")
        
        # Графики сравнения
        st.divider()
        st.subheader("📊 Визуальное сравнение")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            fig_comp_cost = px.bar(
                comparison_df,
                x="route",
                y="total_cost",
                color="total_risk",
                title="Стоимость маршрутов",
                labels={"route": "Маршрут", "total_cost": "Стоимость (₽)", "total_risk": "Риск"}
            )
            st.plotly_chart(fig_comp_cost, use_container_width=True)
        
        with col_comp2:
            fig_comp_time = px.bar(
                comparison_df,
                x="route",
                y="estimated_hours",
                title="Время в пути",
                labels={"route": "Маршрут", "estimated_hours": "Время (ч)"}
            )
            st.plotly_chart(fig_comp_time, use_container_width=True)

# ============================================
# СТРАНИЦА ПРОГНОЗА СПРОСА
# ============================================

elif st.session_state.current_page == "demand":
    st.markdown('<p class="main-header">📈 Прогноз спроса на перевозки</p>', unsafe_allow_html=True)
    
    st.info("ℹ️ Прогноз основан на машинном обучении с учётом сезона, погоды и региона")
    
    col_dem1, col_dem2, col_dem3 = st.columns(3)
    
    with col_dem1:
        dem_month = st.selectbox("Месяц", range(1, 13), format_func=lambda x: datetime(2024, x, 1).strftime("%B"))
    
    with col_dem2:
        dem_temp = st.slider("Температура (°C)", -50, 40, 0)
    
    with col_dem3:
        dem_remote = st.checkbox("Труднодоступный регион")
    
    if st.button("🔮 Рассчитать прогноз спроса", type="primary", use_container_width=True):
        demand = predict_demand(dem_month, 0, dem_temp, 1 if dem_remote else 0)
        
        st.subheader("Результат прогноза")
        
        col_dem_res1, col_dem_res2 = st.columns(2)
        
        with col_dem_res1:
            st.metric("Прогноз спроса", f"{demand:.0f} заявок/день")
        
        with col_dem_res2:
            trend = "📈 Высокий" if demand > 120 else "📊 Средний" if demand > 80 else "📉 Низкий"
            st.metric("Тренд", trend)
        
        # Рекомендации
        st.divider()
        st.subheader("💡 Рекомендации")
        
        if demand > 120:
            st.success("✅ Высокий спрос — рекомендуется увеличить парк ТС на 20-30%")
        elif demand > 80:
            st.info("ℹ️ Средний спрос — текущие мощности достаточны")
        else:
            st.warning("⚠️ Низкий спрос — возможно сокращение парка или перераспределение ресурсов")
        
        # График сезонности
        st.divider()
        st.subheader("📅 Сезонность спроса")
        
        months = range(1, 13)
        demands = [predict_demand(m, 0, dem_temp, 1 if dem_remote else 0) for m in months]
        
        fig_season = px.line(
            x=months,
            y=demands,
            title="Прогноз спроса по месяцам",
            labels={"x": "Месяц", "y": "Спрос (заявок/день)"}
        )
        fig_season.update_xaxes(tickmode='array', tickvals=list(months), 
                               ticktext=[datetime(2024, m, 1).strftime("%b") for m in months])
        st.plotly_chart(fig_season, use_container_width=True)

# ============================================
# СТРАНИЦА DASHBOARD
# ============================================

elif st.session_state.current_page == "dashboard":
    st.markdown('<p class="main-header">📊 Dashboard | Аналитика логистики</p>', unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("ℹ️ Авторизуйтесь для доступа к персональному Dashboard")
        st.stop()
    
    # Загрузка данных
    dashboard_data = get_user_calculations(st.session_state.user_data["user_id"])
    
    if len(dashboard_data) == 0:
        st.info("📭 Нет данных для аналитики. Выполните расчёты!")
        st.stop()
    
    # KPI метрики
    st.subheader("🎯 KPI показатели")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.metric("Всего расчётов", len(dashboard_data))
    with col_kpi2:
        st.metric("Общая стоимость", f"{dashboard_data['total_cost'].sum():,.0f} ₽")
    with col_kpi3:
        avg_risk = dashboard_data['total_risk'].mean()
        risk_emoji = "🔴" if avg_risk >= 0.6 else "🟡" if avg_risk >= 0.3 else "🟠" if avg_risk >= 0.1 else "🟢"
        st.metric("Средний риск", f"{risk_emoji} {avg_risk*100:.1f}%")
    with col_kpi4:
        st.metric("Среднее расстояние", f"{dashboard_data['distance_km'].mean():,.0f} км")
    
    st.divider()
    
    # Графики
    col_dash1, col_dash2 = st.columns(2)
    
    with col_dash1:
        st.subheader("Топ-5 маршрутов по стоимости")
        top_routes = dashboard_data.groupby('route')['total_cost'].sum().nlargest(5).reset_index()
        fig_top = px.bar(top_routes, x='route', y='total_cost', 
                        title="Самые дорогие маршруты",
                        labels={"route": "Маршрут", "total_cost": "Стоимость (₽)"})
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col_dash2:
        st.subheader("Распределение по типам грузов")
        cargo_dist = dashboard_data['cargo_type'].value_counts().reset_index()
        fig_cargo = px.pie(cargo_dist, values='count', names='cargo_type',
                          title="Структура грузов")
        st.plotly_chart(fig_cargo, use_container_width=True)
    
    st.divider()
    
    col_dash3, col_dash4 = st.columns(2)
    
    with col_dash3:
        st.subheader("Динамика расчётов по времени")
        dashboard_data['date'] = pd.to_datetime(dashboard_data['created_at']).dt.date
        time_series = dashboard_data.groupby('date').size().reset_index(name='count')
        fig_time = px.line(time_series, x='date', y='count',
                          title="Количество расчётов по дням",
                          labels={"date": "Дата", "count": "Расчёты"})
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col_dash4:
        st.subheader("Средняя надбавка по регионам")
        # Упрощённый анализ (в реальном проекте нужна привязка к регионам)
        risk_by_road = dashboard_data.groupby('road_type')['total_risk'].mean().reset_index()
        fig_risk = px.bar(risk_by_road, x='road_type', y='total_risk',
                         title="Риск по типу дорог",
                         labels={"road_type": "Тип дороги", "total_risk": "Средний риск"})
        st.plotly_chart(fig_risk, use_container_width=True)

# ============================================
# СТРАНИЦА ЧАТ-БОТА
# ============================================

elif st.session_state.current_page == "chat":
    st.markdown('<p class="main-header">💬 Чат-бот поддержки</p>', unsafe_allow_html=True)
    
    st.info("ℹ️ Задайте вопрос о системе, логистике или климатических рисках")
    
    # Отображение истории чата
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message chat-user"><strong>Вы:</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message chat-bot"><strong>Бот:</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
    
    # Ввод сообщения
    st.divider()
    
    col_chat1, col_chat2 = st.columns([4, 1])
    
    with col_chat1:
        user_input = st.text_input("Ваш вопрос:", key="chat_input", label_visibility="collapsed")
    
    with col_chat2:
        send_button = st.button("Отправить", use_container_width=True)
    
    # Обработка сообщения
    if send_button and user_input:
        # Добавление сообщения пользователя
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Генерация ответа
        user_input_lower = user_input.lower()
        
        # Поиск ответа в базе знаний
        bot_response = "Я не нашёл ответ на ваш вопрос. Попробуйте задать вопрос иначе или обратитесь к документации."
        
        for key, responses in CHATBOT_RESPONSES.items():
            if key in user_input_lower:
                bot_response = responses
                break
        
        # Добавление ответа бота
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        st.rerun()
    
    # Быстрые вопросы
    st.divider()
    st.subheader("⚡ Быстрые вопросы")
    
    col_q1, col_q2, col_q3 = st.columns(3)
    
    with col_q1:
        if st.button("Как влияет климат на стоимость?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "Как влияет климат на стоимость?"})
            st.session_state.chat_history.append({"role": "bot", "content": CHATBOT_RESPONSES["как влияет климат"]})
            st.rerun()
    
    with col_q2:
        if st.button("Какие типы грузов доступны?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "Какие типы грузов доступны?"})
            st.session_state.chat_history.append({"role": "bot", "content": "Доступны 7 типов грузов: Общий, Температурный, Опасный, Негабаритный, Насыпной, Хрупкий, Животные. Каждый тип имеет свои тарифы и ограничения по весу."})
            st.rerun()
    
    with col_q3:
        if st.button("Что такое труднодоступные регионы?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "Что такое труднодоступные регионы?"})
            st.session_state.chat_history.append({"role": "bot", "content": CHATBOT_RESPONSES["что такое труднодоступные регионы"]})
            st.rerun()

# ============================================
# СТРАНИЦА КАЛЬКУЛЯТОРА ПРИБЫЛИ
# ============================================

elif st.session_state.current_page == "profit":
    st.markdown('<p class="main-header">💰 Калькулятор прибыли</p>', unsafe_allow_html=True)
    
    st.info("ℹ️ Рассчитайте прибыль от перевозки с учётом всех расходов")
    
    # Параметры
    col_prof1, col_prof2 = st.columns(2)
    
    with col_prof1:
        revenue = st.number_input("Доход от клиента (₽)", min_value=0.0, value=300000.0, step=1000.0)
        fuel_cost_per_liter = st.number_input("Стоимость топлива (₽/л)", min_value=0.0, value=60.0, step=1.0)
    
    with col_prof2:
        additional_costs = st.number_input("Доп. расходы (зарплата, амортизация, и т.д.) (₽)", 
                                          min_value=0.0, value=50000.0, step=1000.0)
        driver_salary = st.number_input("Зарплата водителя за рейс (₽)", min_value=0.0, value=30000.0, step=1000.0)
    
    # Расчёт на основе последнего расчёта
    if len(st.session_state.calculation_history) > 0:
        last_calc = st.session_state.calculation_history[-1]
        
        st.divider()
        st.subheader("📊 Расчёт на основе последнего маршрута")
        st.caption(f"Маршрут: {last_calc['route']}")
        
        # Расход топлива
        vehicle = next((v for k, v in VEHICLE_TYPES.items() if v["name"] == last_calc["vehicle_type"]), None)
        if vehicle:
            fuel_consumption = vehicle["fuel_consumption"]  # л/100км
            fuel_needed = (last_calc["distance_km"] / 100) * fuel_consumption
            fuel_cost = fuel_needed * fuel_cost_per_liter
            
            total_cost_with_expenses = last_calc["total_cost"] + fuel_cost + additional_costs + driver_salary
            
            profit_data = calculate_profit(revenue, total_cost_with_expenses, 0)
            
            # Отображение результатов
            st.divider()
            st.subheader("📈 Результаты расчёта")
            
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            
            with col_res1:
                st.metric("Доход", f"{revenue:,.0f} ₽")
            with col_res2:
                st.metric("Расходы", f"{total_cost_with_expenses:,.0f} ₽")
            with col_res3:
                profit_class = "profit-positive" if profit_data["is_profitable"] else "profit-negative"
                st.markdown(f'<div class="metric-card"><h3>Прибыль</h3><p class="{profit_class}">{profit_data["profit"]:,.0f} ₽</p></div>', 
                          unsafe_allow_html=True)
            with col_res4:
                margin_class = "profit-positive" if profit_data["profit_margin"] > 15 else "profit-negative"
                st.markdown(f'<div class="metric-card"><h3>Маржа</h3><p class="{margin_class}">{profit_data["profit_margin"]}%</p></div>', 
                          unsafe_allow_html=True)
            
            # Детализация расходов
            st.divider()
            st.subheader("📋 Детализация расходов")
            
            expenses_df = pd.DataFrame({
                "Статья расходов": ["Климатическая стоимость", "Топливо", "Доп. расходы", "Зарплата водителя"],
                "Сумма (₽)": [
                    last_calc["total_cost"],
                    fuel_cost,
                    additional_costs,
                    driver_salary
                ]
            })
            
            st.dataframe(expenses_df, use_container_width=True, hide_index=True)
            
            # Рекомендации
            st.divider()
            st.subheader("💡 Рекомендации")
            
            if profit_data["is_profitable"]:
                if profit_data["profit_margin"] > 25:
                    st.success("✅ Отличная маржа! Рекомендуется принимать заказ.")
                elif profit_data["profit_margin"] > 15:
                    st.info("ℹ️ Хорошая маржа. Заказ выгоден.")
                else:
                    st.warning("⚠️ Маржа на грани рентабельности. Рассмотрите возможность повышения цены.")
            else:
                st.error("❌ Заказ убыточен! Не рекомендуется к принятию.")
                st.info(f"Для безубыточности необходимо увеличить цену на {abs(profit_data['profit']):,.0f} ₽")

# ============================================
# ФУТЕР
# ============================================

st.divider()
st.caption("Climate-TMS Pro © 2026 | Магистерская диссертация | 09.04.03 — Прикладная информатика | Все права защищены")