import requests

NOTIFY_ENDPOINT = "https://sebastianknoll.net/api/?api=optipus&token=h3s9bgi8wsf3h10d2wp0xcff00paquituiedifjp0&id=4"


def notify(numberOfExperiments, r2, q2, repScore, responseName, factorSet, optimum, 
           mainEffects, interactionEffects, quadraticEffects, sendSMS=False):

    r = requests.post(NOTIFY_ENDPOINT, json={
        "sms": sendSMS,
        "numberOfExperiments": numberOfExperiments,
        "stats": {"R2": r2, "Q2": q2, "RepScore": repScore},
        "response": responseName,
        "factorSet": factorSet.serialize(),
        "optimum": optimum,
        "mainEffects": mainEffects,
        "interactionEffects": interactionEffects,
        "quadraticEffects": quadraticEffects
    })

    return r.status_code