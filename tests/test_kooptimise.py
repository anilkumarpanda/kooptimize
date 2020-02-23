import pytest
from kooptimize.korules import get_rules_for_individual

def test_get_rule_for_individual():
    ind = [1,0,1,0]
    rule_dict = {0:"age <= 70",1:"ltv<=90",
    3:"no_of_primary_accts <= 50",4:"credit_history_lenght >= 3"}
    result_dict = get_rules_for_individual(ind)
    expected_dict = {0:"age <= 70",3:"no_of_primary_accts <= 50"}

    assert result_dict == expected_dict