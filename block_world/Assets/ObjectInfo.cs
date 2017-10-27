using System;
using System.Collections;
using System.Net;
using System.Collections.Generic;
using UnityEngine;

namespace Utilities
{
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
}
