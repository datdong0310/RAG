import requests

from config import TEACHER_BASE_URL, HEADERS


def main():
    url = f"{TEACHER_BASE_URL}/competition/evaluate"

    print("[EVALUATE] url =", url)

    try:
        res = requests.post(
            url,
            headers=HEADERS,
            timeout=300
        )

        print("[EVALUATE] status_code =", res.status_code)
        print("[EVALUATE] response =", res.text)

    except requests.exceptions.ReadTimeout:
        print("[EVALUATE] timeout, nhưng Teacher có thể vẫn đang chấm.")

    except Exception as e:
        print("[EVALUATE] error =", e)


if __name__ == "__main__":
    main()