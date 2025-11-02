/**
 * TypeScript Code Generation Test
 * Test: Type-safe state management with generics and discriminated unions
 */

// Type definitions
type Action<T extends string, P = undefined> = P extends undefined
  ? { type: T }
  : { type: T; payload: P };

// State actions
type TodoAction =
  | Action<'ADD_TODO', { id: string; text: string }>
  | Action<'REMOVE_TODO', { id: string }>
  | Action<'TOGGLE_TODO', { id: string }>
  | Action<'SET_FILTER', { filter: TodoFilter }>;

type TodoFilter = 'all' | 'active' | 'completed';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
  createdAt: Date;
}

interface TodoState {
  todos: Todo[];
  filter: TodoFilter;
}

// Reducer with full type safety
function todoReducer(state: TodoState, action: TodoAction): TodoState {
  switch (action.type) {
    case 'ADD_TODO':
      return {
        ...state,
        todos: [
          ...state.todos,
          {
            id: action.payload.id,
            text: action.payload.text,
            completed: false,
            createdAt: new Date(),
          },
        ],
      };

    case 'REMOVE_TODO':
      return {
        ...state,
        todos: state.todos.filter((todo) => todo.id !== action.payload.id),
      };

    case 'TOGGLE_TODO':
      return {
        ...state,
        todos: state.todos.map((todo) =>
          todo.id === action.payload.id
            ? { ...todo, completed: !todo.completed }
            : todo
        ),
      };

    case 'SET_FILTER':
      return {
        ...state,
        filter: action.payload.filter,
      };

    default:
      // Exhaustive check - TypeScript will error if we miss a case
      const _exhaustive: never = action;
      return state;
  }
}

// Selectors with generic constraints
function selectFilteredTodos(state: TodoState): Todo[] {
  switch (state.filter) {
    case 'active':
      return state.todos.filter((todo) => !todo.completed);
    case 'completed':
      return state.todos.filter((todo) => todo.completed);
    default:
      return state.todos;
  }
}

// Generic utility function
function createStore<S, A>(
  reducer: (state: S, action: A) => S,
  initialState: S
) {
  let state = initialState;
  const listeners: Array<(state: S) => void> = [];

  return {
    getState: (): S => state,
    dispatch: (action: A): void => {
      state = reducer(state, action);
      listeners.forEach((listener) => listener(state));
    },
    subscribe: (listener: (state: S) => void): (() => void) => {
      listeners.push(listener);
      return () => {
        const index = listeners.indexOf(listener);
        if (index > -1) listeners.splice(index, 1);
      };
    },
  };
}

// Usage example
const store = createStore(todoReducer, { todos: [], filter: 'all' });

store.subscribe((state) => {
  console.log('State updated:', state);
});

store.dispatch({
  type: 'ADD_TODO',
  payload: { id: '1', text: 'Learn TypeScript' },
});

store.dispatch({
  type: 'TOGGLE_TODO',
  payload: { id: '1' },
});

export { todoReducer, createStore, selectFilteredTodos };
export type { Todo, TodoState, TodoAction };
