import './App.css';
import Dropdown from './components/dropDown';
import Home from './components/Home';
import About from './components/About'
import { BrowserRouter, Route, Routes, Link } from 'react-router-dom';
import React from 'react'
import Draft from './components/Draft';

function App() {
  return (
    <React.StrictMode>
    <div className="app">
      <div className='banner'>
        <h1>WELCOME HONORED GUEST</h1>
      </div>
      <BrowserRouter>
        <div className='nav'>
          <div className ='navItem'>
            <Link to='/'>
              Home
            </Link>
          </div>
          <div className='navItem'>
            <Link to='/About'>
              About
            </Link>
          </div>
          <div className='navItem'>
            <Link to='/Draft'>
              Draft
            </Link>
          </div>
        </div>
        <Routes>
          <Route 
            exact
            path='/'
            element={<Home/>}
            />
            <Route
              exact
              path='/About'
              element={<About/>}
            />
            <Route
              exact
              path='/Draft'
              element={<Draft/>}
            />
        </Routes>
      </BrowserRouter>
      
    </div></React.StrictMode>
  );
}

export default App;

/*

    <Dropdown>
      <Dropdown.Button>Navigate</Dropdown.Button>
      <Dropdown.Content>
        <Dropdown.List>
          <Dropdown.Item to='/'>Home</Dropdown.Item>
          <Dropdown.Item to='/test'>Test</Dropdown.Item> 
        </Dropdown.List>
      </Dropdown.Content>
    </Dropdown>
*/