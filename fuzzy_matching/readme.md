# Fuzzy Matching
Description for `fuzzy_matching` folder

1. dist_*.py: 
    Distance functions for fuzzy matching. You can register your own distance function with decorator `@register_dist_adaptor('FUNCTION NAME')`
2. measurer_*.py:
    Provide basic methods for calculating distances (e.g. dot-product, caching...)