using UnityEngine;
using System.Collections;

public class RenderDepth : MonoBehaviour {
	Material depth_mat;
	Shader DepthShader;

	[Range(0.0f, 3)]
	public float depthLevel = 0.5f;
	public bool highPrecisionDepth = true;
	public bool DepthOn = true;

	void Start() {
		Setup ();
	}
		
	public void Setup () {
		DepthShader = Resources.Load("RenderDepth") as Shader;
		if (DepthShader == null)
			Debug.Log ("Can't find depth shader");
		else {
			depth_mat = new Material (DepthShader);
			depth_mat.name = "Depth shader material";
			GetComponent<Camera> ().depthTextureMode = DepthTextureMode.Depth;
		}
	}

	// Update is called once per frame
	void Update () {
		if(DepthOn)
			GetComponent<Camera> ().depthTextureMode = DepthTextureMode.Depth;
		else
			GetComponent<Camera> ().depthTextureMode = DepthTextureMode.None;
	}

	// Called by the camera to apply the image effect
	void OnRenderImage (RenderTexture source, RenderTexture destination){
		if (DepthOn) {
			//mat is the material containing your shader
			depth_mat.SetFloat ("_DepthLevel", depthLevel);
			if (highPrecisionDepth)
				depth_mat.SetInt ("_HighPrecisionDepth", 1);
			else
				depth_mat.SetInt ("_HighPrecisionDepth", 0);
			Graphics.Blit (source, destination, depth_mat);
		} else
			Graphics.Blit (source, destination);
	}
}