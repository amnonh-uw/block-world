﻿using System.Collections;
using System.Net;
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

	ColorGenerator colorGenerator;

    // Use this for initialization
    void Start()
    {
		//
		// Make main camera show depth!
		//
		RenderDepth RenderScript = Camera.main.gameObject.AddComponent<RenderDepth> ();
		if (RenderScript == null) {
			Debug.Log ("failed to add RenderDepth script to main cam");
		}
		RenderScript.DepthOn = false;

		colorGenerator = new ColorGenerator ();
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

	void PositionMainCamera(GameObject finger)
	{
		Camera.main.transform.position = new Vector3 (0f, 2f, finger.transform.position.z * 2f);
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
		PositionMainCamera (finger);
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

		PositionMainCamera (fing);

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
		cam_controller.transform.localEulerAngles = Vector3.zero;
		cam_controller.transform.localPosition = new Vector3 (distance, 0f, 0f);
		cam_controller.SetActive (false);
		cam.targetTexture = new RenderTexture(p.cameraWidth, p.cameraHeight, 24);
		cam.transform.localEulerAngles = Vector3.zero;
	
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

					// Debug.LogFormat ("listenerAction {0}", listenerAction);
					// if (listenerArgs != null)
					// 	Debug.LogFormat ("listenerArgs {0}", listenerArgs);
					// else
					//	Debug.Log ("no args");

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

					if (listenerAction != null && listenerAction != "" && !lookForAction) {
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
		public bool highprecision;
        public Vector3 finger_pos;
		public Vector3 finger_rot;
        public Vector3 target_pos;
		public Vector3 target_rot;
		public bool collision;
    }

 	byte[] ScreenShot(Camera cam)
	{
		cam.gameObject.SetActive (true);
		RenderTexture currentRT = RenderTexture.active;
		RenderTexture.active = cam.targetTexture;
		cam.Render();
			
		Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
		image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
		image.Apply();
		RenderTexture.active = currentRT;

		byte[] buff = image.EncodeToPNG();
		Destroy (image);
		cam.gameObject.SetActive (false);
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
			RenderDepth rd = depthcam.gameObject.GetComponent<RenderDepth>();
			r.highprecision = rd.highPrecisionDepth;
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
