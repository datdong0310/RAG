import time
import requests

from config import TEACHER_BASE_URL, HEADERS


def main():
    url = f"{TEACHER_BASE_URL}/competition/result"

    for i in range(60):
        print(f"\n[RESULT] check {i + 1}/60")

        try:
            res = requests.get(
                url,
                headers=HEADERS,
                timeout=20
            )

            print("[RESULT] status_code =", res.status_code)
            print("[RESULT] response =", res.text)

            if res.ok:
                data = res.json()
                if data.get("status") == "completed":
                    print("[RESULT] completed")
                    break

        except Exception as e:
            print("[RESULT] error =", e)

        time.sleep(5)


if __name__ == "__main__":
    main()