import numpy as np 
import cv2 
import argparse
import pickle
import os 
from time import time
import matplotlib.pyplot as plt

from utils import * 
import pdb 


def set_arguments(parser): 
    """
    Set the command line arguments for the SFM (Structure from Motion) program.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        None
    """

    #directory stuff
    parser.add_argument('--data_dir',action='store',type=str,default='./assets/assignment2/Benchmarking_Camera_Calibration_2008',dest='data_dir',
                        help='root directory containing input data (default: ../data/)') 
    parser.add_argument('--dataset',action='store',type=str,default='fountain-P11',dest='dataset',
                        help='name of dataset (default: fountain-P11)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext', 
                        help='comma seperated string of allowed image extensions (default: jpg,png)') 
    parser.add_argument('--out_dir',action='store',type=str,default='./assets/assignment2/results/',dest='out_dir',
                        help='root directory to store results in (default: ../results/)') 

    #matching parameters
    parser.add_argument('--features',action='store',type=str,default='SIFT',dest='features',
                        help='[SIFT] Feature algorithm to use (default: SIFT)')
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use (default: BFMatcher)') 
    parser.add_argument('--cross_check',action='store',type=bool,default=True,dest='cross_check',
                        help='[True|False] Whether to cross check feature matching or not (default: True)') 

    #epipolar geometry parameters
    parser.add_argument('--calibration_mat',action='store',type=str,default='benchmark',
                        dest='calibration_mat',help='[benchmark|lg_g3] type of intrinsic camera to use (default: benchmark)')
    parser.add_argument('--fund_method',action='store',type=str,default='FM_RANSAC',
                        dest='fund_method',help='method to estimate fundamental matrix (default: FM_RANSAC)')
    parser.add_argument('--outlier_thres',action='store',type=float,default=.9,
                        dest='outlier_thres',help='threhold value of outlier to be used in fundamental matrix estimation (default: 0.9)')
    parser.add_argument('--fund_prob',action='store',type=float,default=.9,dest='fund_prob',
                        help='confidence in fundamental matrix estimation required (default: 0.9)')
    
    #PnP parameters
    parser.add_argument('--pnp_method',action='store',type=str,default='SOLVEPNP_DLS',
                        dest='pnp_method',help='[SOLVEPNP_DLS|SOLVEPNP_EPNP|..] method used for PnP estimation, see OpenCV doc for more options (default: SOLVEPNP_DLS')
    parser.add_argument('--pnp_prob',action='store',type=float,default=.99,dest='pnp_prob',
                        help='confidence in PnP estimation required (default: 0.99)')
    parser.add_argument('--reprojection_thres',action='store',type=float,default=8.,
                        dest='reprojection_thres',help='reprojection threshold in PnP estimation (default: 8.)')

    #misc
    parser.add_argument('--plot_error',action='store',type=bool,default=False,dest='plot_error')


def post_process_args(opts): 
    """
    Post-processes the command line arguments.

    Args:
        opts (argparse.Namespace): The parsed command line arguments.

    Returns:
        None
    """
    opts.fund_method = getattr(cv2, opts.fund_method)
    opts.ext = opts.ext.split(',')


class Camera(object): 
    """
    Represents a camera in a 3D scene.

    Attributes:
        R (numpy.ndarray): The rotation matrix of the camera.
        t (numpy.ndarray): The translation vector of the camera.
        ref (str): The reference frame of the camera.
    """

    def __init__(self, R, t, ref): 
        self.R = R 
        self.t = t 
        self.ref = ref

class Match(object):
    """
    Represents a set of matches between two images.

    Attributes:
        matches (list): List of matches between the images.
        img1pts (list): List of points in image 1 corresponding to the matches.
        img2pts (list): List of points in image 2 corresponding to the matches.
        img1idx (list): List of indices of matched points in image 1.
        img2idx (list): List of indices of matched points in image 2.
        mask (list): List indicating which matches are valid (True) or invalid (False).
    """

    def __init__(self, matches, img1pts, img2pts, img1idx, img2idx, mask):
        self.matches = matches
        self.img1pts, self.img2pts = img1pts, img2pts
        self.img1idx, self.img2idx = img1idx, img2idx
        self.mask = mask

class SFM(object): 
    def __init__(self, opts): 
        self.opts = opts
        self.point_cloud = np.zeros((0,3))

        #setting up directory stuff..
        self.images_dir = os.path.join(opts.data_dir,opts.dataset, 'images')
        self.feat_dir = os.path.join(opts.data_dir, opts.dataset, 'features', opts.features)
        self.matches_dir = os.path.join(opts.data_dir, opts.dataset, 'matches', opts.matcher)
        self.out_cloud_dir = os.path.join(opts.out_dir, opts.dataset, 'point-clouds')
        self.out_err_dir = os.path.join(opts.out_dir, opts.dataset, 'errors')

        #output directories
        if not os.path.exists(self.out_cloud_dir): 
            os.makedirs(self.out_cloud_dir)

        if (opts.plot_error is True) and (not os.path.exists(self.out_err_dir)): 
            os.makedirs(self.out_err_dir)

        self.image_names = [x.split('.')[0] for x in sorted(os.listdir(self.images_dir)) \
                            if x.split('.')[-1] in opts.ext]

        #setting up shared parameters for the pipeline
        self.image_data, self.matches_data, errors = {}, {}, {}
        self.matcher = getattr(cv2, opts.matcher)(crossCheck=opts.cross_check)

        if opts.calibration_mat == 'benchmark': 
            self.K = np.array([[2759.48,0,1520.69],[0,2764.16,1006.81],[0,0,1]])
        elif opts.calibration_mat == 'lg_g3': 
            self.K = np.array([[3.97*320, 0, 320],[0, 3.97*320, 240],[0,0,1]])
        else: 
            raise NotImplementedError
        
    def load_features(self, name): 
        """
        Load keypoints and descriptors from pickle files.

        Args:
            name (str): The name of the feature set.

        Returns:
            tuple: A tuple containing the keypoints and descriptors.
        """
        with open(os.path.join(self.feat_dir,'kp_{}.pkl'.format(name)),'rb') as f: 
            kp = pickle.load(f)
        kp = deserialize_keypoints(kp)

        with open(os.path.join(self.feat_dir,'desc_{}.pkl'.format(name)),'rb') as f: 
            desc = pickle.load(f)

        return kp, desc

    def load_matches(self, name1, name2): 
            """
            Load matches from a pickle file.

            Args:
                name1 (str): Name of the first image.
                name2 (str): Name of the second image.

            Returns:
                list: List of matches.
            """
            with open(os.path.join(self.matches_dir,'match_{}_{}.pkl'.format(name1,name2)),'rb') as f: 
                matches = pickle.load(f)
            matches = deserialize_matches(matches)
            return matches

    def get_aligned_matches(self,kp1,desc1,kp2,desc2,matches):
            """
            Get aligned matches between two sets of keypoints.

            Args:
                kp1 (list): List of keypoints from image 1.
                desc1 (list): List of descriptors from image 1.
                kp2 (list): List of keypoints from image 2.
                desc2 (list): List of descriptors from image 2.
                matches (list): List of matches between keypoints.

            Returns:
                img1pts (ndarray): Array of image coordinates of matched keypoints from image 1.
                img2pts (ndarray): Array of image coordinates of matched keypoints from image 2.
                img1idx (ndarray): Array of indices of matched keypoints from image 1.
                img2idx (ndarray): Array of indices of matched keypoints from image 2.
            """
            img1idx = np.array([m.queryIdx for m in matches])
            img2idx = np.array([m.trainIdx for m in matches])

            #filtering out the keypoints that were matched. 
            kp1_ = (np.array(kp1))[img1idx]
            kp2_ = (np.array(kp2))[img2idx]

            #retreiving the image coordinates of matched keypoints
            img1pts = np.array([kp.pt for kp in kp1_])
            img2pts = np.array([kp.pt for kp in kp2_])

            return img1pts, img2pts, img1idx, img2idx

    def baseline_pose_estimation(self, name1, name2):
            """
            Estimates the baseline pose between two images.

            Args:
                name1 (str): Name of the first image.
                name2 (str): Name of the second image.

            Returns:
                tuple: A tuple containing the rotation matrix (R) and translation vector (t).
            """
            kp1, desc1 = self.load_features(name1)
            kp2, desc2 = self.load_features(name2)  

            matches = self.load_matches(name1, name2)
            matches = sorted(matches, key = lambda x:x.distance)

            img1pts, img2pts, img1idx, img2idx = self.get_aligned_matches(kp1,desc1,kp2,desc2,matches)
            
            F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,ransacReprojThreshold=opts.outlier_thres,confidence=opts.fund_prob)
            mask = mask.astype(bool).flatten()

            E = self.K.T.dot(F.dot(self.K))
            _,R,t,_ = cv2.recoverPose(E,img1pts[mask],img2pts[mask],self.K)

            self.image_data[name1] = [np.eye(3,3), np.zeros((3,1)), np.ones((len(kp1),))*-1]
            self.image_data[name2] = [R,t,np.ones((len(kp2),))*-1]

            self.matches_data[(name1,name2)] = [matches, img1pts[mask], img2pts[mask], img1idx[mask],img2idx[mask]]

            return R,t

    def triangulate_two_views(self, name1, name2): 
        """
        Triangulates the 3D coordinates of matched points between two views.

        Args:
            name1 (str): Name of the first image view.
            name2 (str): Name of the second image view.

        Returns:
            None
        """

        def triangulation(img1pts, img2pts, R1, t1, R2, t2): 
            """
            Perform triangulation to estimate the 3D coordinates of points in the scene.

            Args:
                img1pts (numpy.ndarray): 2D image points in the first image.
                img2pts (numpy.ndarray): 2D image points in the second image.
                R1 (numpy.ndarray): Rotation matrix of the first camera.
                t1 (numpy.ndarray): Translation vector of the first camera.
                R2 (numpy.ndarray): Rotation matrix of the second camera.
                t2 (numpy.ndarray): Translation vector of the second camera.

            Returns:
                numpy.ndarray: 3D coordinates of the triangulated points.

            """
            img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
            img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

            img1ptsNorm = (np.linalg.inv(self.K).dot(img1ptsHom.T)).T
            img2ptsNorm = (np.linalg.inv(self.K).dot(img2ptsHom.T)).T

            img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
            img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

            pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),
                                            img1ptsNorm.T,img2ptsNorm.T)
            pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

            return pts3d

        def update_3D_reference(ref1, ref2, img1idx, img2idx, upp_limit, low_limit=0): 

            ref1[img1idx] = np.arange(upp_limit) + low_limit
            ref2[img2idx] = np.arange(upp_limit) + low_limit

            return ref1, ref2

        R1, t1, ref1 = self.image_data[name1]
        R2, t2, ref2 = self.image_data[name2]

        _, img1pts, img2pts, img1idx, img2idx = self.matches_data[(name1,name2)]
        
        new_point_cloud = triangulation(img1pts, img2pts, R1, t1, R2, t2)
        self.point_cloud = np.concatenate((self.point_cloud, new_point_cloud), axis=0)

        ref1, ref2 = update_3D_reference(ref1, ref2, img1idx, img2idx,new_point_cloud.shape[0],
                                        self.point_cloud.shape[0]-new_point_cloud.shape[0])
        self.image_data[name1][-1] = ref1 
        self.image_data[name2][-1] = ref2 

    def trangulate_new_view(self, name): 
        """
        Triangulates new view based on matches with previous views.

        Args:
            name (str): Name of the new view.

        Returns:
            None
        """
        for prev_name in self.image_data.keys(): 
            if prev_name != name: 
                kp1, desc1 = self.load_features(prev_name)
                kp2, desc2 = self.load_features(name)  

                prev_name_ref = self.image_data[prev_name][-1]
                matches = self.load_matches(prev_name,name)
                matches = [match for match in matches if prev_name_ref[match.queryIdx] < 0]

                if len(matches) > 0: 
                    # TODO: Process the new view
                    pass
                else: 
                    print('skipping {} and {}'.format(prev_name, name))
        
    def new_view_pose_estimation(self, name): 
        """
        Estimates the pose (rotation and translation) of a new view based on 2D-3D point correspondences.

        Args:
            name (str): The name of the new view.

        Returns:
            None
        """
        
        def find_2D3D_matches(): 
            """
            Finds 2D-3D point correspondences between the new view and existing views.

            Returns:
                pts3d (numpy.ndarray): Array of 3D points.
                pts2d (numpy.ndarray): Array of 2D points.
                ref_len (int): Number of reference keypoints.
            """
            
            matcher_temp = getattr(cv2, opts.matcher)()
            kps, descs = [], []
            for n in self.image_names: 
                if n in self.image_data.keys():
                    kp, desc = self.load_features(n)

                    kps.append(kp)
                    descs.append(desc)
            
            matcher_temp.add(descs)
            matcher_temp.train()

            kp, desc = self.load_features(name)

            matches_2d3d = matcher_temp.match(queryDescriptors=desc)

            #retrieving 2d and 3d points
            pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
            for m in matches_2d3d: 
                train_img_idx, desc_idx, new_img_idx = m.imgIdx, m.trainIdx, m.queryIdx
                point_cloud_idx = self.image_data[self.image_names[train_img_idx]][-1][desc_idx]
                
                #if the match corresponds to a point in 3d point cloud
                if point_cloud_idx >= 0: 
                    new_pt = self.point_cloud[int(point_cloud_idx)]
                    pts3d = np.concatenate((pts3d, new_pt[np.newaxis]),axis=0)

                    new_pt = np.array(kp[int(new_img_idx)].pt)
                    pts2d = np.concatenate((pts2d, new_pt[np.newaxis]),axis=0)

            return pts3d, pts2d, len(kp)

        pts3d, pts2d, ref_len = find_2D3D_matches()
        _, R, t, _ = cv2.solvePnPRansac(pts3d[:,np.newaxis],pts2d[:,np.newaxis],self.K,None,
                            confidence=self.opts.pnp_prob,flags=getattr(cv2,self.opts.pnp_method),
                            reprojectionError=self.opts.reprojection_thres)
        R,_=cv2.Rodrigues(R)
        self.image_data[name] = [R,t,np.ones((ref_len,))*-1]

    def generate_ply(self, filename):
        
        def get_colors(): 
            colors = np.zeros_like(self.point_cloud)
            
            for k in self.image_data.keys(): 
                _, _, ref = self.image_data[k]
                kp, desc = self.load_features(k)
                kp = np.array(kp)[ref>=0]
                image_pts = np.array([_kp.pt for _kp in kp])

                image = cv2.imread(os.path.join(self.images_dir, k+'.jpg'))[:,:,::-1]

                colors[ref[ref>=0].astype(int)] = image[image_pts[:,1].astype(int),
                                                        image_pts[:,0].astype(int)]
            
            return colors

        colors = get_colors()
        pts2ply(self.point_cloud, colors, filename)

    def compute_reprojection_error(self, name): 
        """
        Computes the reprojection error for a given image and also visualize the reprojection as PNG/JPG plot.

        Parameters:
        - name (str): The name of the image.

        Returns:
        - err (float): The average reprojection error.
        """

        # TODO: Reprojection error calculation
        R, t, ref = self.image_data[name]
        kp, desc = self.load_features(name)
        err = 0

        # TODO: PLOT here
        if self.opts.plot_error: 
            fig,ax = plt.subplots()
            image = cv2.imread(os.path.join(self.images_dir, name+'.jpg'))[:,:,::-1]
            # ax = draw_correspondences(image, img_pts, reproj_pts, ax)
            ax.set_title('reprojection error = {}'.format(err))
            fig.savefig(os.path.join(self.out_err_dir, '{}.png'.format(name)))
            plt.close(fig)
            
        return err
        
    def run(self):
        """
        Runs the structure from motion algorithm.

        This method performs the following steps:
        1. Performs baseline pose estimation for the first two images.
        2. Performs baseline triangulation for the first two images.
        3. Generates a 3D point cloud and evaluates reprojection error for the first two images.
        4. Performs pose estimation, triangulation, and reprojection error evaluation for the remaining images.
        5. Calculates the mean reprojection error for all images.

        Returns:
            None
        """
        name1, name2 = self.image_names[0], self.image_names[1]

        total_time, errors = 0, []

        t1 = time()
        self.baseline_pose_estimation(name1, name2)
        t2 = time()
        this_time = t2-t1
        total_time += this_time
        print('Baseline Cameras {0}, {1}: Pose Estimation [time={2:.3}s]'.format(name1, name2, this_time))

        self.triangulate_two_views(name1, name2)
        t1 = time()
        this_time = t1-t2
        total_time += this_time
        print('Baseline Cameras {0}, {1}: Baseline Triangulation [time={2:.3}s]'.format(name1, name2, this_time))

        views_done = 2 

        #3d point cloud generation and reprojection error evaluation
        self.generate_ply(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

        err1 = self.compute_reprojection_error(name1)
        err2 = self.compute_reprojection_error(name2)
        errors.append(err1)
        errors.append(err2)

        print('Camera {}: Reprojection Error = {}'.format(name1, err1))
        print('Camera {}: Reprojection Error = {}'.format(name2, err2))

        for new_name in self.image_names[2:]: 

            #new camera registration
            t1 = time()
            self.new_view_pose_estimation(new_name)
            t2 = time()
            this_time = t2-t1
            total_time += this_time
            print('Camera {0}: Pose Estimation [time={1:.3}s]'.format(new_name, this_time))

            #triangulation for new registered camera
            self.trangulate_new_view(new_name)
            t1 = time()
            this_time = t1-t2
            total_time += this_time
            print('Camera {0}: Triangulation [time={1:.3}s]'.format(new_name, this_time))

            #3d point cloud update and error for new camera
            views_done += 1 
            self.generate_ply(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

            new_err = self.compute_reprojection_error(new_name)
            errors.append(new_err)
            print('Camera {}: Reprojection Error = {}'.format(new_name, new_err))

        mean_error = sum(errors) / float(len(errors))
        print('Reconstruction Completed: Mean Reprojection Error = {2} [t={0:.6}s], Results stored in {1}'.format(total_time, self.opts.out_dir, mean_error))
        

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    set_arguments(parser)
    opts = parser.parse_args()
    post_process_args(opts)
    opts.plot_error = True
    
    sfm = SFM(opts)
    sfm.run()