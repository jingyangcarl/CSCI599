import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const container = document.getElementById( 'container' );
let renderer, stats, gui;
let scene, camera, controls, cube;
let isinitialized = false;

function initScene() {
	scene = new THREE.Scene();
	camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
	
	renderer = new THREE.WebGLRenderer();
	renderer.setSize( window.innerWidth, window.innerHeight );
	container.appendChild( renderer.domElement );

	controls = new OrbitControls( camera, renderer.domElement );
	controls.minDistance = 2;
	controls.maxDistance = 10;
	controls.addEventListener( 'change', function() { renderer.render( scene, camera ); });

	// the loading of the object is asynchronous
	let loader = new OBJLoader();
	loader.load( 
		// resource URL
		'../assets/cube.obj', 
		// called when resource is loaded
		function ( object ) {
			cube = object.children[0];
			cube.material = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
			cube.position.set( 0, 0, 0 );
			cube.name = "cube";
			scene.add( cube );
		},
		// called when loading is in progresses
		function ( xhr ) {
			console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
		},
		// called when loading has errors
		function ( error ) {
			console.log( 'An error happened' + error);
		}
	);
	
	camera.position.z = 5;
}

function initSTATS() {
	stats = new Stats();
	stats.showPanel( 0 );
	container.appendChild( stats.dom );
}

function initGUI() {
	if (!isinitialized) {
		gui = new GUI();
		cube = scene.getObjectByName( "cube" );
		gui.add( cube.position, 'x', -1, 1 );
		gui.add( cube.position, 'y', -1, 1 );
		gui.add( cube.position, 'z', -1, 1 );
		isinitialized = true;
	}
}

function animate() {
	requestAnimationFrame( animate );

	cube = scene.getObjectByName( "cube" );
	if (cube) {
		cube.rotation.x += 0.01;
		cube.rotation.y += 0.01;
		initGUI(); // initialize the GUI after the object is loaded
	}

	renderer.render( scene, camera );
	stats.update();
}

function onWindowResize() {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight );
};

window.addEventListener( 'resize', onWindowResize, false );

initScene();
initSTATS();
animate();
