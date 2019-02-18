def manual_rules(toked_can, dist):
    while len(toked_can) and toked_can[0] in ['and', 'or', ',','.']:
        dist *= 0.5
        toked_can = toked_can[1:]
    if len(toked_can) > 1 and toked_can[1] in [',','.']:
        dist *= 0.5
        toked_can = toked_can[2:]
    while len(toked_can) and toked_can[-1] in ['the', 'of', ',', '.', ':', 'to', 'and', 'in', 'this', 'that', 'or']:
        dist *= 0.5
        toked_can = toked_can[:-1]
    if len(toked_can) > 1 and toked_can[-2] in [',','.']:
        dist *= 0.5
        toked_can = toked_can[:-2]
    return dist