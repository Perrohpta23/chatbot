import sqlite3, os

db_path = os.path.join(os.getcwd(), "chat.db")
if not os.path.exists(db_path):
    print("⚠️ No se encontró chat.db en esta carpeta.")
else:
    con = sqlite3.connect(db_path)
    print("📄 Columnas en la tabla conversation:\n")
    for row in con.execute("PRAGMA table_info(conversation)"):
        print(row)
    con.close()