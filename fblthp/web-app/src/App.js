import './App.css';
import Dropdown from './components/dropDown';
import Home from './components/Home';
import Test from './components/test'
import { BrowserRouter, Route, Routes } from 'react-router-dom';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Dropdown>
          <Dropdown.Button>Open</Dropdown.Button>
          <Dropdown.Content>
            <Dropdown.List>
              <Dropdown.Item to='/'>Home</Dropdown.Item>
              <Dropdown.Item to='/test'>Test</Dropdown.Item> 
            </Dropdown.List>
          </Dropdown.Content>
        </Dropdown>
        <Routes>
          <Route 
            exact
            path='/'
            element={<Home/>}
            />
            <Route
              exact
              path='/test'
              element={<Test/>}
            />
        </Routes>
      </BrowserRouter>
      
    </div>
  );
}

export default App;
