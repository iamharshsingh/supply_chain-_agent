import joblib
import numpy as np
from langchain.tools import tool

# Load trained model
model = joblib.load("Model/supply_chain_model.pkl")

@tool
def analyze_supply_chain(
    product_type: str,
    price: float,
    availability: float,
    stock_levels: float,
    lead_times: float,
    order_quantities: float,
    production_volumes: float,
    manufacturing_costs: float,
    defect_rates: float,
    shipping_times: float
) -> str:
    """
    Forecast product demand and detect supply chain risks using operational metrics.
    """

    try:
        # Convert product_type to dummy variables (based on training)
        product_type = product_type.lower()

        product_type_haircare = 1 if product_type == "haircare" else 0
        product_type_skincare = 1 if product_type == "skincare" else 0
        # cosmetics → baseline (0,0)

        features = [
            price,
            availability,
            stock_levels,
            lead_times,
            order_quantities,
            production_volumes,
            manufacturing_costs,
            defect_rates,
            shipping_times,
            product_type_haircare,
            product_type_skincare
        ]

        arr = np.array(features).reshape(1, -1)

        prediction = model.predict(arr)[0]

        # simple risk analysis
        insights = []

        if stock_levels < 30 and lead_times > 10:
            insights.append("⚠️ High stockout risk detected")

        if availability < 20:
            insights.append("⚠️ Product availability is critically low")

        if defect_rates > 0.3:
            insights.append("⚠️ High defect rate may impact supply chain efficiency")

        if manufacturing_costs > 70:
            insights.append("⚠️ Manufacturing costs are unusually high")

        if not insights:
            insights.append("✅ Supply chain appears stable")

        return (
            f"📊 Predicted Products Sold: {round(prediction,2)}\n\n"
            + "\n".join(insights)
        )

    except Exception as e:
        return f"Error in analysis: {str(e)}"


forecast_tool = analyze_supply_chain