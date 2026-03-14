# login.py

def login(username, password):
    if username == "admin" and password == "1234":
        return "Login successful!"
    else:
        return "Login failed!"

if __name__ == "__main__":
    user = input("Enter username: ")
    pwd = input("Enter password: ")
    print(login(user, pwd))