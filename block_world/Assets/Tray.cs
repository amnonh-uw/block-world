using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.IO;
using System.Text;
using System.Collections.Generic;
using UnityEngine.Assertions;
using UnityEngine;
using Utilities;

public class Tray : MonoBehaviour
{
	Params p;
	public List<GameObject> ObjList = new List<GameObject>();
	public GameObject finger;
	public GameObject target;

	public GameObject leftcam_controller = null;
	public Camera leftcam = null;
	public GameObject rightcam_controller = null;
	public Camera rightcam = null;
	public GameObject centercam_controller = null;
	public Camera centercam = null;

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
   
    bool objectsDropping;
    public bool Ready;
	bool Resetting;

    Listener listener;
    bool takeCameraShot;

    KeyCode Forward = KeyCode.W;
    KeyCode Left = KeyCode.A;
    KeyCode Right = KeyCode.D;
    KeyCode Back = KeyCode.S;
    float Speed = 0.1f;

	public bool CollisionHappened;

	int CommandCounter;

	ColorGenerator colorGenerator;

    // Use this for initialization
    void Start()
    {
		colorGenerator = new ColorGenerator ();
		p = new Params () ;
		ParametersChanged ();
		finger = null;
		target = null;
		CommandCounter = 0;

        listener = new Listener();
        if(!listener.Start()) {
			Debug.Log ("can't listen to port");
			Application.Quit ();
		}

		objectsDropping = false;
		Ready = true;

		Application.runInBackground = true;
    }

	void OnApplicationQuit()
	{
		listener.Stop ();
	}

	private void NoResponse()
	{
		takeCameraShot = false;
		listener.Response.ContentType = "";
		listener.Response.StatusCode = 200;
		listener.Response.StatusDescription = "OK";
		listener.Response.ContentLength64 = 0;
		listener.Response.OutputStream.Close ();
	}

    public void BoolResponse(bool b) {
            ResponseBool r = new ResponseBool();
            r.b = b;
            JsonResponse (JsonUtility.ToJson (r));
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
		objectsDropping = false;
		CollisionHappened = false;
    }
		
	private void ParametersChanged()
	{
		MinDropLength = -0.5f * p.TrayLength + p.RimWidth;
		MaxDropLength = 0.5f * p.TrayLength - p.RimWidth;
		MinDropWidth = -0.5f * p.TrayWidth + p.RimWidth;
		MaxDropWidth = 0.5f * p.TrayWidth - p.RimWidth;
		DropHeight = p.ObjMaxSize * 4.0f;

		PositionMainCamera ();
	}

	private void Initialize()
	{
		objectsDropping = false;
		Ready = false;
		takeCameraShot = false;
		Resetting = false;
		NumObjects = p.MaxObjects;

        if (centercam == null)
            CreateCameras ();

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

	void PositionMainCamera()
	{
		Camera.main.transform.position = new Vector3 (0f, 2f, p.fingerDistFromTray - 1.6f);
		Camera.main.transform.eulerAngles = new Vector3 (45f, 0f, 0f);
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
		r.material.color = colorGenerator.RandomColor ();
		
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
			objectsDropping = true;
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
		return(CreateRandomFinger (MinDropLength,
								   MaxDropLength,
								   p.TrayHeight + p.RimHeight,
			                       p.TrayHeight + p.RimHeight + p.fingerMaxHeight,
								   -0.5f * p.TrayWidth - trayDistance,
								   fingerSize));
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
		targ.transform.localScale = new Vector3(p.TargetSize, p.TargetSize, p.TargetSize);

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
		cam_controller.transform.localEulerAngles = Vector3.zero;
		cam_controller.transform.localPosition = new Vector3 (distance, 0f, 0f);
		cam_controller.SetActive (false);
		cam.targetTexture = new RenderTexture(p.cameraWidth, p.cameraHeight, 24);
		cam.transform.localEulerAngles = Vector3.zero;
		cam_controller.SetActive (true);
	
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
		centercam.gameObject.AddComponent<ImageSynthesis> ();
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

        if (objectsDropping && !Resetting)
        {
			foreach (GameObject obj in ObjList) {
				Vector3 vec = obj.GetComponent<Rigidbody> ().velocity;

				if (p.Nonzero (vec))
					return;
			}
				
            if (!Ready)
            {
                target = CreateRandomTarget();
				objectsDropping = false;
				Ready = true;
                takeCameraShot = true;
                return;
            }
        }

		if (Ready && !Resetting)
        {
            string action, args, body;

			listener.GetAction(out action, out args, out body);

			// if (body != null)
			//	Debug.LogFormat ("Request {0}", body);
			
			if (action == null)
				return;

			// Debug.LogFormat ("Command {0} args {1}", action, args);

        
            if (action != null) {
                CommandCounter += 1;
                // if (CommandCounter % 100 == 0)
                //    Debug.LogFormat ("Total Memory use {0}", System.GC.GetTotalMemory (true));
            }

            if (action == "clear_tray")
            {
                ClearTray ();
				takeCameraShot = true;
                return;
            }

            if (action == "set_finger")
            {
                ObjectInfo info = new ObjectInfo (args, finger);
                SetFinger (info);
                takeCameraShot = true;
                return;
            }

            if (action == "set_target")
            {
                ObjectInfo info = new ObjectInfo (args, target);
                SetTarget (info);
                takeCameraShot = true;
                return;
            }

            if (action == "add_object")
            {
                ObjectInfo info = new ObjectInfo (args);
                AddObject (info);
                takeCameraShot = true;
                return;
            }

            if (action == "move_finger")
            {
                Pose p = new Pose (args);

                if (CollisionCheck(finger, p)) {
                    finger.transform.position += p.position;
                    finger.transform.Rotate (p.rotation);
                }
                takeCameraShot = true;
                return;
            }

            if (action == "check_occupied") {
                Pose p = new Pose (args);
                p.position += finger.transform.position;
               
                BoolResponse (OccupancyCheck (p, finger.transform.localScale));
                return;
            }

            if (action == "move_cams")
            {
                Pose p = new Pose (args);

                if (CollisionCheck(Camera.main.gameObject, p)) {
                    Camera.main.transform.position += p.position;
                    Camera.main.transform.Rotate (p.rotation);
                }

                takeCameraShot = true;
                return;
            }

            if (action == "quit") {
                NoResponse ();
                Application.Quit ();
                return;
            }

            if (p.Set(action, args)) {
                ParametersChanged ();
                NoResponse ();
                return;
            }
						
            if (action == "reset")
            {
                Reset();
                Initialize();
                return;
            }

            if (action == "get_json_params") {
                JsonResponse (p.ToJson ());
                return;
            }

            if (action == "get_json_objectinfo") {
                ObjectsInfo oi = new ObjectsInfo (this);
                JsonResponse (oi.ToJson ());
                return;
            }

            if (action != null && action != "") {
                Debug.LogFormat ("Unknown commmand {0}", action);
                NoResponse ();
                return;
            }
        }
    }

	bool CollisionCheck(GameObject g, Pose p)
	{
		Vector3 direction = p.position;
		float distance = p.position.magnitude;

		if (Physics.Raycast (g.transform.position, direction, distance)) {
			CollisionHappened = true;
			return false;
		}
		return true;
	}

    bool OccupancyCheck(Pose p, Vector3 size) {
		Vector3 halfExtents = size * 0.5f;
		Collider[]  c = Physics.OverlapBox(p.position, halfExtents, Quaternion.identity, -1, QueryTriggerInteraction.Ignore);

        return c.Length != 0;
    }

    [System.Serializable]
    public class ResponseBool
    {
        public bool b;
    }

    [System.Serializable]
    public class Response
    {
        public byte[] leftcam;
        public byte[] rightcam;
		public byte[] centercam;
		public byte[] depthcam;
		public byte[] multichanneldepthcam;
		public byte[] normalcam;
        public Vector3 finger_pos;
		public Vector3 finger_rot;
		public Vector3 finger_screen_pos;
        public Vector3 target_pos;
		public Vector3 target_rot;
		public Vector3 target_screen_pos;
		public bool collision;
    }

 	byte[] ScreenShot(Camera cam)
	{
		// cam.gameObject.SetActive (true);
		RenderTexture currentRT = RenderTexture.active;
		RenderTexture.active = cam.targetTexture;
		cam.Render();

		Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
		image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
		image.Apply();
		RenderTexture.active = currentRT;

		byte[] buff = image.EncodeToPNG();
		Destroy (image);
		// cam.gameObject.SetActive (false);
		return(buff);
	}

	void JsonResponse(string json)
	{
		listener.Response.ContentType = "application/json";
		listener.Response.StatusCode = 200;
		listener.Response.StatusDescription = "OK";
		byte[] buffer = Encoding.UTF8.GetBytes(json);
		listener.Response.ContentLength64 = buffer.Length;
		listener.Response.OutputStream.Write(buffer, 0, buffer.Length);
		listener.Response.OutputStream.Close();
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
			ImageSynthesis center_is = centercam.gameObject.GetComponent<ImageSynthesis> ();
			r.depthcam = center_is.Encode ("_depth");
			if (r.depthcam.Length == 0)
				Debug.Log ("Error: failed to encode _depth");
			r.multichanneldepthcam = center_is.Encode ("_depthmulti");
			if (r.multichanneldepthcam.Length == 0)
				Debug.Log ("Error: failed to encode _depthmulti");
			r.normalcam = center_is.Encode ("_normals");
			if (r.normalcam.Length == 0)
				Debug.Log ("Error: failed to encode _normals");
			
			r.finger_pos = finger.transform.position;
			r.finger_rot = finger.transform.rotation.eulerAngles;
			r.finger_screen_pos = centercam.WorldToScreenPoint (finger.transform.position);
			r.target_pos = target.transform.position;
			r.target_rot = target.transform.rotation.eulerAngles;
			r.target_screen_pos = centercam.WorldToScreenPoint (target.transform.position);
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
