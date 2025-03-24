# SafeWheels - Guidelines for Claude

## Project Structure
- Main code in `safewheels/` package
- Scripts in `scripts/` directory
- Config files in `config/` directory

## Code Style Guidelines
- **Formatting**: 4 spaces for indentation, no tabs
- **Imports**: Group imports: stdlib, third-party, local (separated by newline)
- **Type Hints**: Use when beneficial for clarity
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **DocStrings**: Use for modules and public functions/classes
- **Error Handling**: Use specific exceptions with helpful messages
- **Constants**: Uppercase with underscores
- **Line Length**: Maximum 120 characters
- **Comments**: Only for complex logic or non-obvious decisions
- **Functions**: Keep focused, under 50 lines when possible