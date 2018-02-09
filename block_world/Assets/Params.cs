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
	    public float StereoDistance = 0.1f;

	    public int cameraWidth = 1024;
	    public int cameraHeight = 768;
	
	    public bool Set (string name, string value, Tray tray)
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
	            tray.bottomLength  = System.Convert.ToSingle (value);
	            return true;
	
	        case "tray_height":
	            tray.bottomHeight = System.Convert.ToSingle (value);
	            return true;
	
	        case "tray_width":
	            tray.bottomWidth = System.Convert.ToSingle (value);
	            return true;
	
	        case "rim_height":
	            tray.rimHeight = System.Convert.ToSingle (value);
	            return true;
	
	        case "rim_width":
	            tray.rimWidth = System.Convert.ToSingle (value);
	            return true;
	
	        case "obj_min_size":
	            tray.minObjectSize = System.Convert.ToSingle (value);
	            return true;
	
	        case "obj_max_size":
	            tray.maxObjectSize = System.Convert.ToSingle (value);
	            return true;
	
	        case "max_objects":
	            tray.maxObjects = System.Convert.ToInt32 (value);
	            return true;
	
	        case "stereo_distance":
	            StereoDistance = System.Convert.ToSingle (value);
	            return true;
	        
	        }
	        return false;
	    }
    }
}
