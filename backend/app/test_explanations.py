
from explanations import generate_counterfactuals


if __name__ == "__main__":
    result = generate_counterfactuals(
        professional_fee=200.0,
        facility_fee=100.0,
        service_category="office visit",
        total_cfs=3,
    )

    print(result)
