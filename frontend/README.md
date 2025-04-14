# Career Matcher Frontend

A modern React application for job matching and career insights.

## Features

- Upload PDF resumes for job matching
- View job matches with similarity scores
- Get detailed candidate evaluations
- Identify strengths and areas for improvement
- Responsive design for all devices

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The application will run on [http://localhost:3000](http://localhost:3000).

## Project Structure

```
frontend/
├── public/              # Static files
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/           # Page components
│   ├── services/        # API services
│   ├── types/           # TypeScript type definitions
│   ├── utils/           # Utility functions
│   ├── App.tsx          # Main application component
│   └── index.tsx        # Application entry point
├── package.json         # Project dependencies
└── tsconfig.json        # TypeScript configuration
```

## Development

- The application uses TypeScript for type safety
- Material-UI for the component library
- React Router for navigation
- Axios for API requests

## Building for Production

To create a production build:

```bash
npm run build
```

The build artifacts will be stored in the `build/` directory.

## Testing

To run tests:

```bash
npm test
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 