import numpy as np 
import cv2 
import pdb

def serialize_keypoints(kp): 
    """Serialize list of keypoint objects so it can be saved using pickle
    
    Args: 
    kp: List of keypoint objects 
    
    Returns: 
    out: Serialized list of keypoint objects"""

    out = []
    for kp_ in kp: 
        temp = (kp_.pt, kp_.size, kp_.angle, kp_.response, kp_.octave, kp_.class_id)
        out.append(temp)

    return out

def deserialize_keypoints(kp): 
    """Deserialize list of keypoint objects so it can be converted back to
    native opencv's format.
    
    Args: 
    kp: List of serialized keypoint objects 
    
    Returns: 
    out: Deserialized list of keypoint objects"""

    out = []
    for point in kp:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5]) 
        out.append(temp)

    return out

def serialize_matches(matches): 
    """Serializes dictionary of matches so it can be saved using pickle
    
    Args: 
    matches: List of matches object
    
    Returns: 
    out: Serialized list of matches object"""

    out = []
    for match in matches: 
        matchTemp = (match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
        out.append(matchTemp)
    return out

def deserialize_matches(matches): 
    """Deserialize dictionary of matches so it can be converted back to 
    native opencv's format. 
    
    Args: 
    matches: Serialized list of matches object
    
    Returns: 
    out: List of matches object"""

    out = []
    for match in matches:
        out.append(cv2.DMatch(match[0],match[1],match[2],match[3])) 
    return out

def pts2ply(pts,colors,filename='out.ply'): 
    """Saves an ndarray of 3D coordinates (in meshlab format)"""

    with open(filename,'w') as f: 
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))
        
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        
        f.write('end_header\n')
        
        #pdb.set_trace()
        colors = colors.astype(int)
        for pt, cl in zip(pts,colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                cl[0],cl[1],cl[2]))

def draw_correspondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
    """
    Draws correspondence between ground truth and reprojected feature point

    Args: 
    ptsTrue, ptsReproj: (n,2) numpy array
    ax: matplotlib axis object
    drawOnly: max number of random points to draw

    Returns: 
    ax: matplotlib axis object
    """
    ax.imshow(img)
    
    # TODO: draw correspondence between ptsTrue and ptsReproj

    return ax