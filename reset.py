import requests

from config import TEACHER_BASE_URL, HEADERS


def main():
    url = f"{TEACHER_BASE_URL}/competition/reset"

    print("[RESET] url =", url)

    try:
        res = requests.post(
            url,
            headers=HEADERS,
            timeout=20
        )

        print("[RESET] status_code =", res.status_code)
        print("[RESET] response =", res.text)

    except Exception as e:
        print("[RESET] error =", e)


if __name__ == "__main__":
    main()