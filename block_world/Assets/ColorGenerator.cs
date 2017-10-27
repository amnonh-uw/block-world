using System;
using System.Collections;
using System.Net;
using System.Collections.Generic;
using UnityEngine;

namespace Utilities
{
	public class ColorGenerator
	{
		List<Color32> colorList;
		System.Random rand;

		public ColorGenerator() {
			rand = new System.Random();
			colorList = new List<Color32>(){
				new Color32(128,0,0,255), // maroon  
				new Color32(139,0,0,255), // dark red    
				new Color32(165,42,42,255), // brown   
				new Color32(178,34,34,255), // firebrick   
				new Color32(220,20,60,255), // crimson 
				// Red is reserved for finger
				// new Color32(255,0,0,255), // red 
				//
				new Color32(255,99,71,255), // tomato  
				new Color32(255,127,80,255), // coral   
				new Color32(205,92,92,255), // indian red  
				new Color32(240,128,128,255), // light coral 
				new Color32(233,150,122,255), // dark salmon 
				new Color32(250,128,114,255), // salmon  
				new Color32(255,160,122,255), // light salmon    
				new Color32(255,69,0,255), // orange red  
				new Color32(255,140,0,255), // dark orange 
				new Color32(255,165,0,255), // orange  
				new Color32(255,215,0,255), // gold    
				new Color32(184,134,11,255), // dark golden rod 
				new Color32(218,165,32,255), // golden rod  
				new Color32(238,232,170,255), // pale golden rod 
				new Color32(189,183,107,255), // dark khaki  
				new Color32(240,230,140,255), // khaki   
				new Color32(128,128,0,255), // olive   
				new Color32(255,255,0,255), // yellow  
				new Color32(154,205,50,255), // yellow green    
				new Color32(85,107,47,255), // dark olive green    
				new Color32(107,142,35,255), // olive drab  
				new Color32(124,252,0,255), // lawn green  
				new Color32(127,255,0,255), // chart reuse 
				new Color32(173,255,47,255), // green yellow    
				new Color32(0,100,0,255), // dark green  
				new Color32(0,128,0,255), // green   
				new Color32(34,139,34,255), // forest green    
				new Color32(0,255,0,255), // lime    
				new Color32(50,205,50,255), // lime green  
				new Color32(144,238,144,255), // light green 
				new Color32(152,251,152,255), // pale green  
				new Color32(143,188,143,255), // dark sea green  
				new Color32(0,250,154,255), // medium spring green 
				new Color32(0,255,127,255), // spring green    
				new Color32(46,139,87,255), // sea green   
				new Color32(102,205,170,255), // medium aqua marine  
				new Color32(60,179,113,255), // medium sea green    
				new Color32(32,178,170,255), // light sea green 
				new Color32(47,79,79,255), // dark slate gray 
				new Color32(0,128,128,255), // teal    
				new Color32(0,139,139,255), // dark cyan   
				new Color32(0,255,255,255), // aqua    
				new Color32(0,255,255,255), // cyan    
				new Color32(224,255,255,255), // light cyan  
				new Color32(0,206,209,255), // dark turquoise  
				new Color32(64,224,208,255), // turquoise   
				new Color32(72,209,204,255), // medium turquoise    
				new Color32(175,238,238,255), // pale turquoise  
				new Color32(127,255,212,255), // aqua marine 
				new Color32(176,224,230,255), // powder blue 
				new Color32(95,158,160,255), // cadet blue  
				new Color32(70,130,180,255), // steel blue  
				new Color32(100,149,237,255), // corn flower blue    
				new Color32(0,191,255,255), // deep sky blue   
				new Color32(30,144,255,255), // dodger blue 
				new Color32(173,216,230,255), // light blue  
				new Color32(135,206,235,255), // sky blue    
				new Color32(135,206,250,255), // light sky blue  
				new Color32(25,25,112,255), // midnight blue   
				new Color32(0,0,128,255), // navy    
				new Color32(0,0,139,255), // dark blue   
				new Color32(0,0,205,255), // medium blue 
				new Color32(0,0,255,255), // blue    
				new Color32(65,105,225,255), // royal blue  
				new Color32(138,43,226,255), // blue violet 
				new Color32(75,0,130,255), // indigo  
				new Color32(72,61,139,255), // dark slate blue 
				new Color32(106,90,205,255), // slate blue  
				new Color32(123,104,238,255), // medium slate blue   
				new Color32(147,112,219,255), // medium purple   
				new Color32(139,0,139,255), // dark magenta    
				new Color32(148,0,211,255), // dark violet 
				new Color32(153,50,204,255), // dark orchid 
				new Color32(186,85,211,255), // medium orchid   
				new Color32(128,0,128,255), // purple  
				new Color32(216,191,216,255), // thistle 
				new Color32(221,160,221,255), // plum    
				new Color32(238,130,238,255), // violet  
				new Color32(255,0,255,255), // magenta / fuchsia   
				new Color32(218,112,214,255), // orchid  
				new Color32(199,21,133,255), // medium violet red   
				new Color32(219,112,147,255), // pale violet red 
				new Color32(255,20,147,255), // deep pink   
				new Color32(255,105,180,255), // hot pink    
				new Color32(255,182,193,255), // light pink  
				new Color32(255,192,203,255), // pink    
				new Color32(250,235,215,255), // antique white   
				new Color32(245,245,220,255), // beige   
				new Color32(255,228,196,255), // bisque  
				new Color32(255,235,205,255), // blanched almond 
				new Color32(245,222,179,255), // wheat   
				new Color32(255,248,220,255), // corn silk   
				new Color32(255,250,205,255), // lemon chiffon   
				new Color32(250,250,210,255), // light golden rod yellow 
				new Color32(255,255,224,255), // light yellow    
				new Color32(139,69,19,255), // saddle brown    
				new Color32(160,82,45,255), // sienna  
				new Color32(210,105,30,255), // chocolate   
				new Color32(205,133,63,255), // peru    
				new Color32(244,164,96,255), // sandy brown 
				new Color32(222,184,135,255), // burly wood  
				new Color32(210,180,140,255), // tan 
				new Color32(188,143,143,255), // rosy brown  
				new Color32(255,228,181,255), // moccasin    
				new Color32(255,222,173,255), // navajo white    
				new Color32(255,218,185,255), // peach puff  
				new Color32(255,228,225,255), // misty rose  
				new Color32(255,240,245,255), // lavender blush  
				new Color32(250,240,230,255), // linen   
				new Color32(253,245,230,255), // old lace    
				new Color32(255,239,213,255), // papaya whip 
				new Color32(255,245,238,255), // sea shell   
				new Color32(245,255,250,255), // mint cream  
				new Color32(112,128,144,255), // slate gray  
				new Color32(119,136,153,255), // light slate gray    
				new Color32(176,196,222,255), // light steel blue    
				new Color32(230,230,250,255), // lavender    
				new Color32(255,250,240,255), // floral white    
				new Color32(240,248,255,255), // alice blue  
				new Color32(248,248,255,255), // ghost white 
				new Color32(240,255,240,255), // honeydew    
				new Color32(255,255,240,255), // ivory   
				new Color32(240,255,255,255), // azure   
				new Color32(255,250,250,255), // snow    
				new Color32(105,105,105,255), // dim gray / dim grey 
				new Color32(128,128,128,255), // gray / grey 
				new Color32(169,169,169,255), // dark gray / dark grey   
				new Color32(192,192,192,255), // silver  
				new Color32(211,211,211,255), // light gray / light grey 
				new Color32(220,220,220,255), // gainsboro   
				new Color32(245,245,245,255), // white smoke 
				new Color32(255,255,255,255) // white   
			};
		}

		public Color RandomColor() {
			//  
			// Color c = Random.ColorHSV(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
			//
			Color32 c = colorList[rand.Next(0, colorList.Count)];

			return (Color) c;
		}
	}
}
