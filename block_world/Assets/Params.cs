using System;
using System.Collections;
using System.Net;
using System.Collections.Generic;
using UnityEngine;

namespace Utilities
{
    [System.Serializable]
    public class Params 
    {
	    public float TrayLength = 1.0f;
	    public float TrayWidth = 0.5f;
	    public float TrayHeight = 0.03f;
	    public float RimHeight = 0.1f;
	    public float RimWidth = 0.01f;
	    public int MaxObjects = 5;
	    public float ObjMinSize = 0.15f;
	    public float ObjMaxSize = 0.3f;
	    public float velEpsilon = 0.15f;
	    public float fingerDistFromTray = 0.0f;
	    public float fingerMaxHeight = 0.8f;
	    public float fingerSize = 0.05f;
	    public float TargetSize = 0.05f;
	    public float StereoDistance = 0.1f;

	    public int cameraWidth = 800;
	    public int cameraHeight = 600;
	
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
	        case "width":
	            cameraWidth = System.Convert.ToInt32 (value);
	            return true;
	
	        case "height":
	            cameraHeight = System.Convert.ToInt32 (value);
	            return true;
	
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
	}
}
