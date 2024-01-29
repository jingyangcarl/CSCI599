import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const container = document.getElementById( 'container' );
let renderer, stats, gui;
let scene, camera, controls, cube;

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
	
	let geometry = new THREE.BoxGeometry( 1, 1, 1 );
	let material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
	cube = new THREE.Mesh( geometry, material );
	scene.add( cube );
	
	camera.position.z = 5;
}

function initSTATS() {
	stats = new Stats();
	stats.showPanel( 0 );
	container.appendChild( stats.dom );
}

function initGUI() {
	gui = new GUI();
	gui.add( cube.position, 'x', -1, 1 );
	gui.add( cube.position, 'y', -1, 1 );
	gui.add( cube.position, 'z', -1, 1 );
}

function animate() {
	requestAnimationFrame( animate );

	cube.rotation.x += 0.01;
	cube.rotation.y += 0.01;

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
initGUI();
animate();