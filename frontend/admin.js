const HttpStatus = { Unauthorized: 401 };
const Field = { Empty: "" };
const YearOfStudy = { First: "1", Second: "2", Third: "3", Fourth: "4" };
const Badge = { Ready: "is_ready" };
const NewStudentId = "";
const EmptySelection = 0;

const login_wrap = document.querySelector("#login_wrap");
const login_form = document.querySelector("#login_form");
const login_error = document.querySelector("#login_error");
const registry = document.querySelector("#registry");
const student_form = document.querySelector("#student_form");
const form_error = document.querySelector("#form_error");
const save_button = document.querySelector("#save_button");
const cancel_edit = document.querySelector("#cancel_edit");
const roster = document.querySelector("#roster");
const attendance = document.querySelector("#attendance");
const sign_out = document.querySelector("#sign_out");

let editing_id = NewStudentId;
let students = [];

function show_registry(visible) {
  login_wrap.hidden = visible;
  registry.hidden = !visible;
}

async function read_json(response) {
  const payload = await response.json();
  if (response.status === HttpStatus.Unauthorized) {
    show_registry(false);
    throw new Error(payload.message || "Sign in again.");
  }
  if (!response.ok) throw new Error(payload.message || "The registry could not save that.");
  return payload;
}

async function refresh() {
  const student_response = await fetch("/api/students");
  students = await read_json(student_response);
  const attendance_response = await fetch("/api/attendance/today");
  const today = await read_json(attendance_response);
  paint_roster();
  paint_attendance(today);
}

function paint_roster() {
  roster.innerHTML = "";
  if (!students.length) {
    roster.innerHTML = '<p class="empty">No photographs yet. Add the first student above.</p>';
    return;
  }
  students.forEach((student) => {
    const article = document.createElement("article");
    article.className = "person";
    const photo = document.createElement("img");
    photo.src = student.photo || "";
    photo.alt = student.name;
    const copy = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = student.name;
    const meta = document.createElement("p");
    meta.textContent = student.details_complete
      ? `${student.index_number} · ${student.program} · ${student.year_label}`
      : "Index number, programme, and year are still open.";
    const badge = document.createElement("span");
    badge.className = student.details_complete ? `badge ${Badge.Ready}` : "badge";
    badge.textContent = student.details_complete ? "Ready to scan" : "Add details";
    copy.append(title, meta, badge);
    const actions = document.createElement("div");
    actions.className = "actions";
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "ghost";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => begin_edit(student));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => remove_student(student));
    actions.append(edit, remove);
    article.append(photo, copy, actions);
    roster.append(article);
  });
}

function paint_attendance(today) {
  if (!today.items.length) {
    attendance.innerHTML = '<p class="empty">No attendance recorded on campus today.</p>';
    return;
  }
  const table = document.createElement("table");
  const head = document.createElement("tr");
  ["Time", "Student", "Index", "Programme", "Year", "Mark"].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    head.append(cell);
  });
  table.append(head);
  today.items.forEach((item) => {
    const row = document.createElement("tr");
    [item.time, item.name, item.index_number, item.program, item.year_label, item.mark].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    table.append(row);
  });
  attendance.innerHTML = "";
  attendance.append(table);
}

function begin_edit(student) {
  editing_id = String(student.id);
  student_form.elements.name.value = student.name;
  student_form.elements.index_number.value = student.index_number;
  student_form.elements.program.value = student.program;
  student_form.elements.year_of_study.value = student.year_of_study || Field.Empty;
  student_form.elements.photo.required = false;
  save_button.textContent = "Save details";
  cancel_edit.hidden = false;
  form_error.textContent = "";
  student_form.scrollIntoView({ behavior: "smooth", block: "start" });
}

function clear_edit() {
  editing_id = NewStudentId;
  student_form.reset();
  student_form.elements.photo.required = true;
  save_button.textContent = "Save student";
  cancel_edit.hidden = true;
  form_error.textContent = "";
}

async function remove_student(student) {
  const confirmed = window.confirm(`Remove ${student.name} from the registry?`);
  if (!confirmed) return;
  const response = await fetch(`/api/students/${student.id}`, { method: "DELETE" });
  await read_json(response);
  if (editing_id === String(student.id)) clear_edit();
  await refresh();
}

login_form.addEventListener("submit", async (event) => {
  event.preventDefault();
  login_error.textContent = "";
  const payload = {
    username: login_form.elements.username.value.trim(),
    password: login_form.elements.password.value,
  };
  const response = await fetch("/api/admin/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (response.status === HttpStatus.Unauthorized) {
    const body = await response.json();
    login_error.textContent = body.message;
    return;
  }
  show_registry(true);
  clear_edit();
  await refresh();
});

sign_out.addEventListener("click", async () => {
  await fetch("/api/admin/logout", { method: "POST" });
  show_registry(false);
});

cancel_edit.addEventListener("click", clear_edit);

student_form.addEventListener("submit", async (event) => {
  event.preventDefault();
  form_error.textContent = "";
  const year = student_form.elements.year_of_study.value;
  const known_years = [YearOfStudy.First, YearOfStudy.Second, YearOfStudy.Third, YearOfStudy.Fourth];
  if (!known_years.includes(year)) {
    form_error.textContent = "Choose a year of study.";
    return;
  }
  const body = new FormData(student_form);
  const creating = editing_id === NewStudentId;
  if (creating && student_form.elements.photo.files.length === EmptySelection) {
    form_error.textContent = "Choose a photograph of the student.";
    return;
  }
  const url = creating ? "/api/students" : `/api/students/${editing_id}`;
  try {
    const response = await fetch(url, { method: "POST", body });
    await read_json(response);
    clear_edit();
    await refresh();
  } catch (error) {
    form_error.textContent = error.message;
  }
});

student_form.elements.photo.required = true;

fetch("/api/admin/session")
  .then((response) => response.json())
  .then((session) => {
    if (!session.signed_in) return;
    show_registry(true);
    return refresh();
  })
  .catch(() => show_registry(false));
