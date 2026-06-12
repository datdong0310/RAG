import requests

from config import TEACHER_BASE_URL, HEADERS, STUDENT_SERVER_URL


def main():
    url = f"{TEACHER_BASE_URL}/competition/register"

    payload = {
        "server_url": STUDENT_SERVER_URL
    }

    print("[REGISTER] url =", url)
    print("[REGISTER] payload =", payload)

    try:
        res = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=20
        )

        print("[REGISTER] status_code =", res.status_code)
        print("[REGISTER] response =", res.text)

    except Exception as e:
        print("[REGISTER] error =", e)


if __name__ == "__main__":
    main()