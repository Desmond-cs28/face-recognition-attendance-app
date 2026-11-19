const LocationVerdict = {
  OnCampus: "on_campus",
  OffCampus: "off_campus",
  Imprecise: "imprecise",
  Missing: "missing",
};
const FaceOutcome = {
  Matched: "matched",
  Unknown: "unknown",
  NoFace: "no_face",
  Incomplete: "incomplete",
};
const RecordState = {
  Recorded: "recorded",
  Already: "already_recorded",
  Blocked: "blocked",
  Incomplete: "incomplete",
};
const AttendanceStatus = { Present: "present", Late: "late" };
const GeoError = { Denied: 1, Unavailable: 2, Timeout: 3 };
const StepState = { Current: "is_current", Done: "is_done", Blocked: "is_blocked" };
const Field = { Empty: "" };
const Dash = "—";
const JpegQuality = 0.85;
const EmptyWidth = 0;
const ClockZone = "Africa/Accra";
const LocationOptions = { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 };
const CameraErrorName = {
  NotAllowed: "NotAllowedError",
  NotFound: "NotFoundError",
  NotReadable: "NotReadableError",
  Security: "SecurityError",
  Overconstrained: "OverconstrainedError",
};
const SecurePage = true;
const CameraAttempts = [
  { video: { facingMode: { ideal: "user" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
  { video: true, audio: false },
];

const video = document.querySelector("#video");
const stage_panel = document.querySelector("#stage_panel");
const stage_note = document.querySelector("#stage_note");
const camera_button = document.querySelector("#camera_button");
const campus_card = document.querySelector("#campus_card");
const campus_title = document.querySelector("#campus_title");
const campus_message = document.querySelector("#campus_message");
const campus_address = document.querySelector("#campus_address");
const check_button = document.querySelector("#check_button");
const scan_button = document.querySelector("#scan_button");
const result_photo = document.querySelector("#result_photo");
const result_kicker = document.querySelector("#result_kicker");
const result_name = document.querySelector("#result_name");
const result_index = document.querySelector("#result_index");
const result_program = document.querySelector("#result_program");
const result_year = document.querySelector("#result_year");
const result_attendance = document.querySelector("#result_attendance");
const result_message = document.querySelector("#result_message");
const step_location = document.querySelector("#step_location");
const step_face = document.querySelector("#step_face");
const step_record = document.querySelector("#step_record");

let latest_fix = null;
let latest_verdict = LocationVerdict.Missing;
let camera_ready = false;
let camera_task = null;
let scan_busy = false;

function show_value(value) {
  if (!value || value === Field.Empty) return Dash;
  return value;
}

function set_step(element, state) {
  element.classList.remove(StepState.Current, StepState.Done, StepState.Blocked);
  if (state) element.classList.add(state);
}

function paint_steps(verdict, recorded) {
  const on_campus = verdict === LocationVerdict.OnCampus;
  set_step(step_location, on_campus ? StepState.Done : StepState.Blocked);
  if (verdict === LocationVerdict.Missing) set_step(step_location, StepState.Current);
  set_step(step_face, camera_ready ? StepState.Current : null);
  if (recorded) {
    set_step(step_face, StepState.Done);
    set_step(step_record, StepState.Done);
    return;
  }
  set_step(step_record, on_campus ? null : StepState.Blocked);
}

function paint_campus(location) {
  latest_verdict = location.verdict || LocationVerdict.Missing;
  campus_title.textContent = "Checking location…";
  campus_card.classList.remove("is_on", "is_off");
  if (location.address) campus_address.textContent = location.address;
  campus_message.textContent = location.message;
  if (location.verdict === LocationVerdict.OnCampus) {
    campus_title.textContent = "On campus";
    campus_card.classList.add("is_on");
  } else if (location.verdict === LocationVerdict.OffCampus) {
    campus_title.textContent = "Outside campus";
    campus_card.classList.add("is_off");
  } else if (location.verdict === LocationVerdict.Imprecise) {
    campus_title.textContent = "Location is too broad";
    campus_card.classList.add("is_off");
  } else {
    campus_title.textContent = "Location needed";
    campus_card.classList.add("is_off");
  }
  const on_campus = latest_verdict === LocationVerdict.OnCampus;
  scan_button.textContent = on_campus ? "Take attendance" : "Scan face";
  scan_button.disabled = scan_busy || !camera_ready;
  paint_steps(latest_verdict, false);
}

function denied_location(error) {
  let message = "Location is unavailable on this device. Attendance needs a campus fix.";
  if (error && error.code === GeoError.Denied) {
    message = "Allow location access so the campus check can run.";
  }
  if (error && error.code === GeoError.Timeout) {
    message = "The location check timed out. Try again in the open, with precise location on.";
  }
  latest_fix = null;
  paint_campus({ verdict: LocationVerdict.Missing, message: message, address: campus_address.textContent });
}

function request_location() {
  campus_title.textContent = "Checking location…";
  campus_message.textContent = "Reading the precise location of this device.";
  if (!navigator.geolocation) {
    denied_location(null);
    return;
  }
  navigator.geolocation.getCurrentPosition(send_fix, denied_location, LocationOptions);
}

async function send_fix(position) {
  latest_fix = {
    latitude: position.coords.latitude,
    longitude: position.coords.longitude,
    accuracy_metres: position.coords.accuracy,
  };
  const response = await fetch("/api/location/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(latest_fix),
  });
  const location = await response.json();
  paint_campus(location);
}

function show_camera_message(message) {
  stage_panel.hidden = false;
  stage_note.textContent = message;
}

function camera_message(error) {
  const name = error && error.name;
  if (name === CameraErrorName.NotAllowed) {
    return "Allow the camera for this site, then press Start camera again.";
  }
  if (name === CameraErrorName.NotFound) {
    return "No camera was found. Open this page on the phone that has the camera.";
  }
  if (name === CameraErrorName.NotReadable) {
    return "The camera is busy in another app. Close that app, then press Start camera.";
  }
  if (name === CameraErrorName.Security) {
    return "Open http://localhost:4721/ in Chrome or Safari so the camera can start.";
  }
  if (name === CameraErrorName.Overconstrained) {
    return "This camera needs a simpler request. Press Start camera again.";
  }
  return "Press Start camera and allow the prompt so your face can appear.";
}

async function open_stream() {
  let last_error = null;
  for (const constraints of CameraAttempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      last_error = error;
      const name = error && error.name;
      if (name === CameraErrorName.NotAllowed || name === CameraErrorName.Security) throw error;
    }
  }
  throw last_error;
}

function attach_stream(stream) {
  video.muted = true;
  video.playsInline = true;
  video.setAttribute("playsinline", "");
  video.srcObject = stream;
  const played = video.play();
  if (played && typeof played.catch === "function") played.catch(() => {});
  camera_ready = true;
  stage_panel.hidden = true;
  scan_button.disabled = !!scan_busy;
  paint_steps(latest_verdict, false);
}

async function open_camera() {
  if (camera_ready) return;
  if (window.isSecureContext !== SecurePage) {
    show_camera_message("Open http://localhost:4721/ in Chrome or Safari so the camera can start.");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    show_camera_message("Use Chrome or Safari. This browser cannot open the camera.");
    return;
  }
  camera_button.disabled = true;
  camera_button.textContent = "Starting camera…";
  try {
    attach_stream(await open_stream());
  } catch (error) {
    camera_button.disabled = false;
    camera_button.textContent = "Start camera";
    show_camera_message(camera_message(error));
  }
}

function start_camera() {
  if (camera_ready || camera_task) return camera_task;
  camera_task = open_camera().finally(() => {
    camera_task = null;
  });
  return camera_task;
}

function frame_payload() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const data_url = canvas.toDataURL("image/jpeg", JpegQuality);
  return data_url.split(",")[1];
}

function attendance_label(result) {
  const attendance = result.attendance;
  if (!attendance) return Dash;
  if (attendance.record_state === RecordState.Recorded && attendance.status === AttendanceStatus.Late) {
    return "Recorded · Late";
  }
  if (attendance.record_state === RecordState.Recorded) return "Recorded · Present";
  if (attendance.record_state === RecordState.Already) return "Already recorded";
  if (attendance.record_state === RecordState.Incomplete) return "Details missing";
  return "Not recorded";
}

function paint_result(result) {
  const student = result.student;
  result_message.textContent = result.message;
  if (!student) {
    result_kicker.textContent = result.outcome === FaceOutcome.NoFace ? "Try again" : "No match";
    result_name.textContent = result.outcome === FaceOutcome.NoFace ? "No face in the frame" : "Unknown face";
    result_photo.hidden = true;
    result_index.textContent = Dash;
    result_program.textContent = Dash;
    result_year.textContent = Dash;
  } else {
    result_kicker.textContent = student.details_complete ? "Student" : "Face on file";
    result_name.textContent = student.name;
    result_index.textContent = show_value(student.index_number);
    result_program.textContent = show_value(student.program);
    result_year.textContent = show_value(student.year_label);
    if (student.photo) {
      result_photo.hidden = false;
      result_photo.src = student.photo;
      result_photo.alt = student.name;
    }
  }
  const label = attendance_label(result);
  result_attendance.textContent = label;
  result_attendance.className = "pill";
  const recorded = result.attendance && (
    result.attendance.record_state === RecordState.Recorded
    || result.attendance.record_state === RecordState.Already
  );
  if (recorded) result_attendance.classList.add("is_recorded");
  if (result.attendance && result.attendance.record_state === RecordState.Blocked) {
    result_attendance.classList.add("is_blocked");
  }
  paint_steps(latest_verdict, recorded);
}

async function take_attendance() {
  if (!camera_ready || video.videoWidth === EmptyWidth) {
    show_camera_message("Press Start camera and wait until your face is inside the oval.");
    return;
  }
  scan_busy = true;
  scan_button.disabled = true;
  scan_button.textContent = "Reading the face…";
  const body = {
    image: frame_payload(),
    latitude: latest_fix ? latest_fix.latitude : null,
    longitude: latest_fix ? latest_fix.longitude : null,
    accuracy_metres: latest_fix ? latest_fix.accuracy_metres : null,
  };
  try {
    const response = await fetch("/api/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (result.location) paint_campus(result.location);
    paint_result(result);
  } catch (error) {
    result_message.textContent = "The scan did not reach the registry. Try once more.";
  }
  scan_busy = false;
  scan_button.disabled = false;
  const on_campus = latest_verdict === LocationVerdict.OnCampus;
  scan_button.textContent = on_campus ? "Take attendance" : "Scan face";
}

check_button.addEventListener("click", request_location);
scan_button.addEventListener("click", take_attendance);
camera_button.addEventListener("click", start_camera);
request_location();
start_camera();
setInterval(() => {
  const clock = new Date().toLocaleTimeString("en-GB", { timeZone: ClockZone, hour: "2-digit", minute: "2-digit" });
  document.querySelector(".place").textContent = `Manya Jorpanya · ${clock}`;
}, 1000);
