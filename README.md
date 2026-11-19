# Palm University College attendance

Students mark attendance by scanning their face. The mark is stored only when the phone’s precise location is inside the Manya Jorpanya campus, at 1 University Avenue Annex.

The registry office matches each photograph to a name, index number, programme, and year of study. Students use the attendance screen only. The registry is a separate sign-in.

## Run

```bash
cd backend
python3 -m pip install -r requirements.txt
python3 app.py
```

Open the attendance screen at `/` and the registry at `/admin`.

Registry sign-in: username `admin`, password `palm2024`. Set `ADMIN_PASSWORD` to change it.

Photographs already in `backend/known_faces` are loaded the first time the registry is empty. Finish their index number, programme, and year on the registry page before a scan can record attendance.

Campus centre: 5.920471, 0.031308. A scan counts only when the reported point, plus its accuracy, sits inside 500 metres of that point.
