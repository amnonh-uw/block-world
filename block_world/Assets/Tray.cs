using System.Collections;
using System.Net;
using System.Threading;
using System.IO;
using System.Text;
using System.Collections.Generic;
using UnityEngine.Assertions;
using UnityEngine;

[System.Serializable]
public class Params 
{
	public float TrayLength = 1.0f;
	public float TrayWidth = 0.5f;
	public float TrayHeight = 0.03f;
	public float RimHeight = 0.1f;
	public float RimWidth = 0.01f;
	public int MaxObjects = 5;
	public float ObjMinSize = 0.05f;
	public float ObjMaxSize = 0.2f;
	public float velEpsilon = 0.15f;
	public float fingerDistFromTray = 0.2f;
	public float fingerMaxHeight = 0.2f;
	public float fingerSize = 0.05f;
	public float TargetSize = 0.05f;
	public float StereoDistance = 0.1f;

	bool Nonzero(float f) {
		return f < -velEpsilon || f > velEpsilon;
	}

	public bool Nonzero(Vector3 vec) {
		return Nonzero (vec.x) || Nonzero (vec.y) || Nonzero(vec.z);
	}
		
	public bool Set (string name, string value)
	{
		switch (name) 
		{
		case "tray_length":
			TrayLength = System.Convert.ToSingle (value);
			return true;

		case "tray_height":
			TrayHeight = System.Convert.ToSingle (value);
			return true;

		case "tray_width":
			TrayWidth = System.Convert.ToSingle (value);
			return true;

		case "rim_height":
			RimHeight = System.Convert.ToSingle (value);
			return true;

		case "rim_width":
			RimWidth = System.Convert.ToSingle (value);
			return true;

		case "obj_min_size":
			ObjMinSize = System.Convert.ToSingle (value);
			return true;

		case "obj_max_size":
			ObjMaxSize = System.Convert.ToSingle (value);
			return true;

		case "max_objects":
			MaxObjects = System.Convert.ToInt32 (value);
			return true;

		case "finger_size":
			fingerSize = System.Convert.ToSingle (value);
			return true;

		case "finger_max_height":
			fingerMaxHeight = System.Convert.ToSingle (value);
			return true;

		case "finger_distance_from_tray":
			fingerDistFromTray = System.Convert.ToSingle (value);
			return true;
		
		case "target_size":
			TargetSize = System.Convert.ToSingle (value);
			return true;

		case "stereo_distance":
			StereoDistance = System.Convert.ToSingle (value);
			return true;
		
		}
		return false;
	}

	public string ToJson() {
		return JsonUtility.ToJson (this);
	}

	public void FromJson(string json) {
		Debug.Log (json);
		JsonUtility.FromJsonOverwrite (json, this);
	}
};

class ElementSplitter {
	string[] elems;
	int index;

	public ElementSplitter(string s) {
		index = 0;
		s = s.Replace (" ", string.Empty);
		s = s.Replace ("(", string.Empty);
		s = s.Replace (")", string.Empty);

 		elems = s.Split (',');
	}

	public float GetNext(float def)
	{
		if (index < elems.Length) {
			string s = elems [index++].Trim();
			if (s == string.Empty)
				return def;
			return System.Convert.ToSingle (s);
		} else
			return def;
	}

	public Vector3 GetNext(Vector3 def) {
		return new Vector3 (GetNext (def.x), GetNext (def.y), GetNext (def.z));
	}

	public Vector4 GetNext(Vector4 def) {
		return new Vector4 (GetNext (def.x), GetNext (def.y), GetNext (def.z), GetNext (def.w));
	}

	public Quaternion GetNext(Quaternion def) {
		return new Quaternion (GetNext (def.x), GetNext (def.y), GetNext (def.z), GetNext (def.w));
	}
		
		
}


[System.Serializable]
public class ObjectInfo
{
	public Vector3 localScale;
	public Vector3 position;
	public Quaternion rotation;
	public Vector4 color;

	public  ObjectInfo(GameObject go) {
		FromWorldTransform (go.transform);
		Renderer r = go.GetComponent<Renderer> ();
		if(r != null)
			color = r.material.color;
	}

	public void SetObject(GameObject go) {
		SetObjectTransform(go);
		go.GetComponent<Renderer> ().material.color = color;
	}

	public void SetObjectTransform(GameObject go) {
		go.transform.position = position;
		go.transform.rotation = rotation;
		go.transform.localScale = localScale;
	}
		
	void Initialize(string s, GameObject go) {
		Vector3 def_position;
		Quaternion def_rotation;
		Vector3 def_localScale;
		Vector4 def_color = Color.green;

		if (go == null) {
			def_position = Vector3.zero;
			def_rotation = Quaternion.identity;
			def_localScale = Vector3.one;
			def_color = Color.green;
		} else {
			def_position = go.transform.position;
			def_rotation = go.transform.rotation;
			def_localScale = go.transform.localScale;
			Renderer r = go.GetComponent<Renderer> ();
			if (r != null)
				def_color = r.material.color;
			else
				def_color = Color.blue;
		}

		ElementSplitter es = new ElementSplitter (s);


		position = es.GetNext (def_position);
		rotation = es.GetNext (def_rotation);
		localScale = es.GetNext (def_localScale);
		color = es.GetNext (def_color);
	}

	public ObjectInfo(string s, GameObject go) {
		Initialize (s, go);
	}

	public ObjectInfo(string s) {
		Initialize (s, null);
	}

	public void FromWorldTransform(Transform t) {
		position = t.position;
		rotation = t.rotation;
		localScale = t.localScale;
	}

	public void Dump() {
		Debug.LogFormat ("position {0}", position);
		Debug.LogFormat ("rotation {0}", rotation);
		Debug.LogFormat ("localScale {0}", localScale);
		Debug.LogFormat ("color {0}", color);
	}
}

[System.Serializable]
class ObjectsInfo {
	public List<ObjectInfo> ObjList = new List<ObjectInfo> ();
	public ObjectInfo finger;
	public ObjectInfo target;
	public ObjectInfo main_camera;

	public ObjectsInfo(Tray t) {
		finger = new ObjectInfo (t.finger);
		target = new ObjectInfo (t.target);
		main_camera = new ObjectInfo (Camera.main.gameObject);

		foreach (GameObject go in t.ObjList)
			ObjList.Add (new ObjectInfo(go));
	}

	public string ToJson() {
		string s = JsonUtility.ToJson (this);
		return s;
	}

	public void Dump() {
		Debug.LogFormat ("finger {0}", finger.ToString ());
		Debug.LogFormat ("target {0}",  target.ToString());
		Debug.LogFormat ("camera {0}",  main_camera.ToString());
		foreach (ObjectInfo info in ObjList)
			Debug.LogFormat ("object {0}",  info.ToString());		
	}
};

public class Tray : MonoBehaviour
{
	Params p;
	public List<GameObject> ObjList = new List<GameObject>();
	public GameObject finger;
	public GameObject target;
	public GameObject leftcam_controller;
	public Camera leftcam;
	public GameObject rightcam_controller;
	public Camera rightcam;
	public GameObject centercam_controller;
	public Camera centercam;
	public GameObject depthcam_controller;
	public Camera depthcam;

	public GameObject tray;
	public GameObject rim1;
	public GameObject rim2;
	public GameObject rim3;
	public GameObject rim4;

    public Material TrayMat;
   
    float MinDropLength;
    float MaxDropLength;
    float MinDropWidth;
    float MaxDropWidth;

	int NumObjects;
    float DropHeight;
    public Material ObjBaseMat;
   
    bool TrackDropping;
    public bool Ready;
	bool Resetting;

    HttpListener listener;
    Thread listenerThread;
    bool exitListener;
    bool takeCameraShot;
    HttpListenerResponse ListenerResponse;
	string listenerAction;
	string listenerArgs;
    Object listenerLock;

    KeyCode Forward = KeyCode.W;
    KeyCode Left = KeyCode.A;
    KeyCode Right = KeyCode.D;
    KeyCode Back = KeyCode.S;
    float Speed = 0.1f;

	public bool CollisionHappened;

	int CommandCounter;

    // Use this for initialization
    void Start()
    {
		p = new Params () ;
		CreateCameras ();
		finger = null;
		target = null;
		CommandCounter = 0;

        listener = new HttpListener();
		string port = System.Environment.GetEnvironmentVariable("PORT");
		if (port == null)
			port = "9000";

		string prefix = "http://*:" + port + "/";
        listener.Prefixes.Add(prefix);
		Debug.Log ("Listening to " + prefix);
        listener.Start();

        listenerThread = new Thread(listenerMain);
        exitListener = false;
        listenerLock = new Object();
        listenerThread.Start();

		TrackDropping = false;
		Ready = true;

		Application.runInBackground = true;
    }

    private void listenerMain()
    {
        while(!exitListener)
        {
            HttpListenerContext context = listener.GetContext();
            HttpListenerRequest request = context.Request;
            ListenerResponse = context.Response;

            if (request.HasEntityBody)
            {
                Stream body = request.InputStream;
                Encoding encoding = request.ContentEncoding;
                StreamReader reader = new StreamReader(body, encoding);


                if (request.ContentType.ToLower() == "application/x-www-form-urlencoded")
                {
                    string s = reader.ReadToEnd();
                    string[] pairs = s.Split('&');

                    lock (listenerLock)
                    {
						listenerAction = null;
						listenerArgs = null;
                        for (int i = 0; i < pairs.Length; i++)
                        {
							string[] items = WWW.UnEscapeURL(pairs[i]).Split(new char[] {'='}, 2);
                            string name = items[0];
							string value = items[1];

							if (name == "command") 
								listenerAction = value;

							if (name == "args") {
								// Debug.LogFormat ("Listener args <{0}>", value);
						listenerArgs = value;
							}
                        }
                    }
                }

                body.Close();
                reader.Close();
            }
        }

        listener.Stop();
    }

	private void NoResponse()
	{
		takeCameraShot = false;
		ListenerResponse.ContentType = "";
		ListenerResponse.StatusCode = 200;
		ListenerResponse.StatusDescription = "OK";
		ListenerResponse.ContentLength64 = 0;
		ListenerResponse.OutputStream.Close ();
	}

	public void ClearObjects() {
		foreach (GameObject obj in ObjList)
			Destroy (obj);
		ObjList.Clear ();
	}

    private void Reset()
    {
		DestroyTray ();
		ClearObjects ();

		if (target != null) {
			Destroy (target);
			target = null;
		}

		if (finger != null) {
			Destroy (finger);
			finger = null;
		}

		Resetting = true;
		Ready = false;
		TrackDropping = false;
		CollisionHappened = false;
    }

	private void InitializeValues () {
		TrackDropping = false;
		Ready = false;
		takeCameraShot = false;
		listenerAction = null;
		Resetting = false;
		NumObjects = p.MaxObjects;

		MinDropLength = -0.5f * p.TrayLength + p.RimWidth;
		MaxDropLength = 0.5f * p.TrayLength - p.RimWidth;
		MinDropWidth = -0.5f * p.TrayWidth + p.RimWidth;
		MaxDropWidth = 0.5f * p.TrayWidth - p.RimWidth;
		DropHeight = p.ObjMaxSize * 4.0f;
	}

	private void Initialize()
	{
		InitializeValues ();
		CreateTray ();
		DropNextObject();
		finger = CreateRandomFinger(p.fingerDistFromTray, p.fingerSize);
	}
		
	private void ClearTray()
	{
		ClearObjects ();
		DestroyTray ();
		CreateTray ();
		if (finger == null)
			finger = CreateRandomFinger(p.fingerDistFromTray, p.fingerSize);
		if (target == null)
			target = CreateRandomTarget();
			
	}

	private void DestroyTray()
	{
		if (tray != null) {
			Destroy (tray);
			tray = null;
		}

		if (rim1 != null) {
			Destroy (rim1);
			rim1 = null;
		}

		if (rim2 != null) {
			Destroy (rim1);
			rim2 = null;
		}

		if (rim3 != null) {
			Destroy (rim1);
			rim3 = null;
		}

		if (rim4 != null) {
			Destroy (rim1);
			rim4 = null;
		}
	}

	private void CreateTray()
	{
        tray = GameObject.CreatePrimitive(PrimitiveType.Cube);
        tray.transform.localScale = new Vector3(p.TrayLength, p.TrayHeight, p.TrayWidth);
        tray.transform.position = new Vector3(0.0f, -0.5f * p.TrayHeight, 0.0f);
        tray.GetComponent<Renderer>().material = TrayMat;
        tray.isStatic = true;
        tray.name = "Tray";

        rim1 = GameObject.CreatePrimitive(PrimitiveType.Cube);
        rim1.transform.localScale = new Vector3(p.TrayLength, p.RimHeight, p.RimWidth);
        rim1.transform.position = new Vector3(0.0f, 0.5f * p.RimHeight, (0.5f * p.TrayWidth) - (0.5f * p.RimWidth));
        rim1.transform.parent = tray.transform;
        rim1.GetComponent<Renderer>().material = TrayMat;
        rim1.isStatic = true;
        rim1.name = "Rim1";

        rim2 = GameObject.CreatePrimitive(PrimitiveType.Cube);
        rim2.transform.localScale = new Vector3(p.TrayLength, p.RimHeight, p.RimWidth);
        rim2.transform.position = new Vector3(0.0f, 0.5f * p.RimHeight, (-0.5f * p.TrayWidth) + (0.5f * p.RimWidth));
        rim2.transform.parent = tray.transform;
        rim2.GetComponent<Renderer>().material = TrayMat;
        rim2.isStatic = true;
        rim2.name = "Rim2";

        rim3 = GameObject.CreatePrimitive(PrimitiveType.Cube);
        rim3.transform.localScale = new Vector3(p.RimWidth, p.RimHeight, p.TrayWidth);
        rim3.transform.position = new Vector3((0.5f * p.TrayLength) - (0.5f * p.RimWidth), 0.5f * p.RimHeight, 0.0f);
        rim3.transform.parent = tray.transform;
        rim3.GetComponent<Renderer>().material = TrayMat;
        rim3.isStatic = true;
        rim3.name = "Rim3";

        rim4 = GameObject.CreatePrimitive(PrimitiveType.Cube);
        rim4.transform.localScale = new Vector3(p.RimWidth, p.RimHeight, p.TrayWidth);
        rim4.transform.position = new Vector3((-0.5f * p.TrayLength) + (0.5f * p.RimWidth), 0.5f * p.RimHeight, 0.0f);
        rim4.transform.parent = tray.transform;
        rim4.GetComponent<Renderer>().material = TrayMat;
        rim4.isStatic = true;
        rim4.name = "Rim4";
    }

	void AddObject(ObjectInfo info)
	{
		GameObject new_object = CreateObject ();

		info.SetObject (new_object);
	}

	GameObject CreateObject()
	{
		GameObject obj = GameObject.CreatePrimitive(PrimitiveType.Cube);

		Material mat = Object.Instantiate(ObjBaseMat);
		obj.GetComponent<Renderer>().material = mat;
		obj.AddComponent<Rigidbody>();
		obj.name = "Object";
		ObjList.Add(obj);

		return obj;
	}

	GameObject CreateRandomObject() 
	{
		GameObject obj = CreateObject ();
	
		Material mat = Object.Instantiate(ObjBaseMat);
		Renderer r = obj.GetComponent<Renderer> ();
		r.material = mat;
		r.material.color = Random.ColorHSV(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

		return obj;	
	}

    void DropObject(float size, float xPos, float zPos)
	{
		GameObject obj = CreateRandomObject ();
        obj.transform.localScale = new Vector3(size, size, size);
        obj.transform.position = new Vector3(xPos, DropHeight, zPos);
   
       	ObjectCollision colscript = obj.AddComponent<ObjectCollision>();
		colscript.trayscript = this;
    }

    public void DropNextObject()
    {
		if (NumObjects-- <= 0 || Resetting)
        {
			// Debug.Log ("Done Dropping");
			TrackDropping = true;
            return;
        }

        float objSize = Random.Range(p.ObjMinSize, p.ObjMaxSize);
        float xPos = Random.Range(MinDropLength + objSize, MaxDropLength - objSize);
        float zPos = Random.Range(MinDropWidth + objSize, MaxDropWidth - objSize);

		DropObject (objSize, xPos, zPos);
    }

	void SetFinger(ObjectInfo info)
	{
		if (finger == null)
			finger = CreateFinger ();

		info.SetObjectTransform (finger);
	}

	GameObject CreateFinger()
	{
		GameObject fing = GameObject.CreatePrimitive(PrimitiveType.Cube);
		fing.GetComponent<Renderer>().material.color = Color.red;
		fing.name = "Finger";

		FingerCollision fingscript = fing.AddComponent<FingerCollision>();
		fingscript.trayscript = this;

		return fing;
	}

	GameObject CreateRandomFinger(float xMin, float xMax, float yMin, float yMax, float zPos, float size)
	{
		GameObject fing = CreateFinger ();

		fing.transform.localScale = new Vector3(size, size, size);
		float xPos = Random.Range (xMin, xMax);
		float yPos = Random.Range (yMin, yMax);
		fing.transform.position = new Vector3(xPos, yPos, zPos);

		return fing;
	}

	GameObject CreateRandomFinger(float trayDistance, float fingerSize)
	{
		return(CreateRandomFinger (MinDropLength,MaxDropLength,
			p.TrayHeight + p.RimHeight, p.TrayHeight + p.RimHeight + p.fingerMaxHeight,
			-0.5f * p.TrayWidth + trayDistance, fingerSize));
	}

	void SetTarget(ObjectInfo info)
	{
		if (target == null)
			target = CreateTarget ();

		info.SetObjectTransform (target);
	}

	GameObject CreateTarget()
	{
		GameObject targ = GameObject.CreatePrimitive(PrimitiveType.Cube);
		targ.GetComponent<Renderer> ().material.color = Color.black;
		targ.name = "Target";

		return (targ);
	}

	GameObject CreateRandomTarget()
	{
		GameObject targ = CreateTarget ();
		targ.transform.localScale = new Vector3(0.03f, p.TargetSize, p.TargetSize);

		Vector3 position;
		Quaternion rotation;

		if (ObjList.Count != 0) {
			RaycastHit hit = RandomTarget ();
			Assert.IsTrue (hit.collider.name == "Object");

			position = hit.point;
			rotation = Quaternion.LookRotation (hit.normal);
			//
			// Don't set the target parent, to allow targets without objects
			//
			// targ.transform.parent = hit.collider.gameObject.transform;
		} else {
			//
			// No objects, just put the target in a random point
			//

			float xPos = Random.Range(MinDropLength + 0.03f, MaxDropLength - 0.03f);
			float zPos = Random.Range(MinDropWidth + p.TargetSize, MaxDropWidth - p.TargetSize);
			float yPos = Random.Range(p.TrayHeight, DropHeight);

			position = new Vector3(xPos, yPos, zPos);
			rotation = Quaternion.identity;
		}
		
		targ.transform.position = position;
		targ.transform.rotation = rotation;

		return (targ);
	}

	GameObject CreateCamera(string name, float distance)
	{
		Camera cam;

		GameObject cam_controller = new GameObject (name);
		cam = cam_controller.AddComponent<Camera> ();
		cam_controller.transform.parent = Camera.main.gameObject.transform;
		cam_controller.transform.localPosition = new Vector3 (distance, 0f, 0f);
		cam_controller.SetActive (false);
		cam.targetTexture = new RenderTexture(cam.pixelWidth, cam.pixelWidth, 24);

		return cam_controller;
	}

	void CreateCameras()
	{
		leftcam_controller = CreateCamera ("Left Camera", -p.StereoDistance / 2f);
		leftcam = leftcam_controller.GetComponent<Camera> ();
		rightcam_controller = CreateCamera ("Right Camera", p.StereoDistance / 2f);
		rightcam = rightcam_controller.GetComponent<Camera> ();
		centercam_controller = CreateCamera ("Center Camera", 0f);
		centercam = centercam_controller.GetComponent<Camera> ();
		depthcam_controller = CreateCamera ("Center Depth Camera", 0f);
		depthcam = depthcam_controller.GetComponent<Camera> ();
		RenderDepth RenderScript = depthcam_controller.AddComponent<RenderDepth> ();
		if (RenderScript == null) {
			Debug.Log ("failed to add RenderDepth script to depth cam");
		}
		RenderScript.Setup ();
	}


    // Update is called once per frame
    void Update()
    {
		Camera cam = Camera.main;

        if (Input.GetKey(Forward))
            cam.transform.position += new Vector3(0, 0, Speed);
        if (Input.GetKey(Left))
            cam.transform.position += new Vector3(-Speed, 0, 0);
        if (Input.GetKey(Right))
            cam.transform.position += new Vector3(Speed, 0, 0);
        if (Input.GetKey(Back))
            cam.transform.position += new Vector3(0, 0, -Speed);

        //float h = Input.GetAxis("Mouse X");
        //float v = Input.GetAxis("Mouse Y");
        //cam.transform.position += new Vector3(Speed * h, 0f, Speed * v);

        if (TrackDropping && !Resetting)
        {
			foreach (GameObject obj in ObjList) {
				Vector3 vec = obj.GetComponent<Rigidbody> ().velocity;

				if (p.Nonzero (vec))
					return;
			}
				
            if (!Ready)
            {
				// Debug.LogFormat ("Ready {0} objects remaining", ObjList.Count);
                target = CreateRandomTarget();
				TrackDropping = false;
				Ready = true;
            }
        }

		if (Ready && !Resetting)
        {
            lock (listenerLock)
            {
				bool lookForAction = false;
                if (listenerAction != null)
                {
					// Debug.LogFormat ("Listener action {0}", listenerAction);
					//
					// actions:
					//	move_finger (inc_pose)
					//	move_cams	(inc_pos)
					//	quit
					//	named parameters
					//	reset
					//	get_json_params
					// 	get_json_objectinfo
					//
					// random scenario:
					//	reset, then sequence of move_finger and move_cams
					//
					// preset scenrio
					//	clear_tray
					//	set_finger
					//	set_target
					//	move_cams
					//	add_object
					//  then sequence of move_finger and move_cams
					//	


					CommandCounter += 1;
					if (CommandCounter % 100 == 0)
						Debug.LogFormat ("Total Memory use {0}", System.GC.GetTotalMemory (true));

					if (listenerAction == "clear_tray")
					{
						ClearTray ();
						takeCameraShot = true;
						listenerAction = null;
					}

					if (listenerAction == "set_finger")
					{
						ObjectInfo info = new ObjectInfo (listenerArgs, finger);
						SetFinger (info);
						takeCameraShot = true;
						listenerAction = null;
					}

					if (listenerAction == "set_target")
					{
						ObjectInfo info = new ObjectInfo (listenerArgs, target);
						SetTarget (info);
						takeCameraShot = true;
						listenerAction = null;
					}

					if (listenerAction == "add_object")
					{
						ObjectInfo info = new ObjectInfo (listenerArgs);
						AddObject (info);
						takeCameraShot = true;
						listenerAction = null;
					}

					if (listenerAction == "move_finger")
					{
						Pose p = new Pose (listenerArgs);

						if (CollisionCheck(finger, p)) {
							finger.transform.position += p.position;
							finger.transform.Rotate (p.rotation);
						}
						takeCameraShot = true;
						listenerAction = null;
					}

					if (listenerAction == "move_cams")
					{
						Pose p = new Pose (listenerArgs);

						if (CollisionCheck(Camera.main.gameObject, p)) {
							Camera.main.transform.position += p.position;
							Camera.main.transform.Rotate (p.rotation);
						}

						takeCameraShot = true;
						listenerAction = null;
					}

					if (listenerAction == "quit") {
						NoResponse ();
						Application.Quit ();
						listenerAction = null;
					}

					if (p.Set(listenerAction, listenerArgs)) {
						NoResponse ();
						listenerAction = null;
					}
						
					if (listenerAction == "wait_for_ready") {
						takeCameraShot = true;
						listenerAction = null;
					}

                    if (listenerAction == "reset")
                    {
                        Reset();
                        Initialize();
						lookForAction = true;
                        listenerAction = "wait_for_ready";
                    }

					if (listenerAction == "get_json_params") {
						JsonResponse (p.ToJson ());
						listenerAction = null;
					}

					if (listenerAction == "get_json_objectinfo") {
						ObjectsInfo oi = new ObjectsInfo (this);
						JsonResponse (oi.ToJson ());
						listenerAction = null;
					}

					if (listenerAction != null && !lookForAction) {
						Debug.LogFormat ("Unknown commmand {0}", listenerAction);
						NoResponse ();
						listenerArgs = null;
					}
                }
            }
        }
    }

	bool CollisionCheck(GameObject g, Pose p)
	{
		Vector3 direction = p.position;
		float distance = p.position.magnitude;

		if (Physics.Raycast (g.transform.position, direction, distance)) {
			// Debug.LogFormat ("Collision check on {0} pose {1} failed", g.name, p.ToString());
			CollisionHappened = true;
			return false;
		}
		// Debug.LogFormat ("Collision check on {0} pose {1} succeeded", g.name, p.ToString());
		return true;
	}

    [System.Serializable]
    public class Response
    {
        public byte[] leftcam;
        public byte[] rightcam;
		public byte[] centercam;
		public byte[] depthcam;
        public Vector3 finger_pos;
		public Vector3 finger_rot;
        public Vector3 target_pos;
		public Vector3 target_rot;
		public bool collision;
    }

 	byte[] ScreenShot(Camera cam)
	{
		RenderTexture currentRT = RenderTexture.active;
		RenderTexture.active = cam.targetTexture;
		cam.Render();
			
		Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
		image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
		image.Apply();
		RenderTexture.active = currentRT;

		byte[] buff = image.EncodeToPNG();
		Destroy (image);
		return(buff);
	}

	void JsonResponse(string json)
	{
		ListenerResponse.ContentType = "application/json";
		ListenerResponse.StatusCode = 200;
		ListenerResponse.StatusDescription = "OK";
		byte[] buffer = Encoding.UTF8.GetBytes(json);
		ListenerResponse.ContentLength64 = buffer.Length;
		ListenerResponse.OutputStream.Write(buffer, 0, buffer.Length);
		ListenerResponse.OutputStream.Close();
	}
   
    void LateUpdate()
    {
		if (takeCameraShot && !Resetting)
        {
            takeCameraShot = false;

            Response r = new Response();
			r.leftcam = ScreenShot (leftcam);
			r.rightcam = ScreenShot (rightcam);
			r.centercam = ScreenShot (centercam);
			r.depthcam = ScreenShot (depthcam);
			r.finger_pos = finger.transform.position;
			r.finger_rot = finger.transform.rotation.eulerAngles;
			r.target_pos = target.transform.position;
			r.target_rot = target.transform.rotation.eulerAngles;
			r.collision = CollisionHappened;

			// Debug.LogFormat ("Colission: {0}", CollisionHappened);

			JsonResponse (JsonUtility.ToJson (r));
        }
	}

 	RaycastHit RandomTarget()
	{
		//
		// Pick random object
		//
		
		int i = Random.Range (0, ObjList.Count);
		GameObject t1 = (GameObject) ObjList [i];


		//
		// Ray from center of the object in random direction hits surface at point
		//


		Vector3 direction1 = Random.insideUnitSphere;
		Ray ray1 = new Ray(t1.transform.position, direction1);

		//
		// Rays from inside the object do not trigger the collider, so we simply reverse the ray
		//
		ray1.origin = ray1.GetPoint(100);
		ray1.direction = -ray1.direction;
		Collider col = t1.GetComponent<Collider> ();
		Assert.IsTrue (col != null);

		RaycastHit hitInfo;
		bool hit;
		hit = col.Raycast (ray1, out hitInfo, 100.0F);
		Assert.IsTrue (hit);
	

		//
		// Ray from finger to point on surface guaranteed to hit an object surface
		//

		Vector3 direction2 = hitInfo.point - finger.transform.position;
		Ray ray2 = new Ray (finger.transform.position, direction2);
		hit = Physics.Raycast (ray2, out hitInfo, 100.0F);
		Assert.IsTrue (hit);

		return hitInfo;
	}
}

class Pose
{
	public Vector3 position;
	public Vector3 rotation;

	public Pose(string s)
	{
		ElementSplitter es = new ElementSplitter (s);

		position = es.GetNext(Vector3.zero);
		rotation = es.GetNext(Vector3.zero);
	}

	public override string ToString()
	{
		return "postion: " + this.position.ToString () + " rotation: " + this.rotation.ToString ();
	}
};