#!/usr/bin/env python3
"""
Extended Language Packs for Code Attractor System
Express/TypeScript, Spring/Java, and more framework patterns
"""

from typing import Dict, List
from dataclasses import dataclass
from code_attractor_system import CodePattern

class LanguagePacks:
    """Extended language and framework pattern collections"""

    @staticmethod
    def get_express_typescript_pack() -> Dict[str, CodePattern]:
        """Express.js + TypeScript patterns"""
        return {
            "express_typed_route": CodePattern(
                name="express_typed_route",
                summary="Type-safe Express route with middleware and validation",
                apis=["express.Router", "Request", "Response", "NextFunction", "body"],
                snippets=[
                    "router.post('/users', async (req: Request, res: Response, next: NextFunction) => {\n"
                    "  try {\n"
                    "    const user: User = req.body;\n"
                    "    const result = await userService.create(user);\n"
                    "    res.status(201).json(result);\n"
                    "  } catch (error) {\n"
                    "    next(error);\n"
                    "  }\n"
                    "});"
                ],
                antipatterns=["missing next()", "untyped request/response", "no error handling"],
                resonance=0.92,
                source="express_typescript_pack"
            ),

            "express_auth_middleware": CodePattern(
                name="express_auth_middleware",
                summary="JWT authentication middleware with TypeScript types",
                apis=["jsonwebtoken", "Request", "Response", "NextFunction", "Authorization"],
                snippets=[
                    "interface AuthRequest extends Request {\n"
                    "  user?: User;\n"
                    "}\n\n"
                    "const authMiddleware = (req: AuthRequest, res: Response, next: NextFunction) => {\n"
                    "  const token = req.headers.authorization?.split(' ')[1];\n"
                    "  if (!token) return res.status(401).json({ error: 'No token' });\n"
                    "  \n"
                    "  try {\n"
                    "    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as JwtPayload;\n"
                    "    req.user = decoded;\n"
                    "    next();\n"
                    "  } catch (error) {\n"
                    "    res.status(401).json({ error: 'Invalid token' });\n"
                    "  }\n"
                    "};"
                ],
                antipatterns=["hardcoded secrets", "missing error handling", "no type safety"],
                resonance=0.89,
                source="express_typescript_pack"
            ),

            "express_error_handler": CodePattern(
                name="express_error_handler",
                summary="Centralized error handling middleware with logging",
                apis=["Error", "Request", "Response", "NextFunction", "logger"],
                snippets=[
                    "interface AppError extends Error {\n"
                    "  statusCode?: number;\n"
                    "  isOperational?: boolean;\n"
                    "}\n\n"
                    "const errorHandler = (err: AppError, req: Request, res: Response, next: NextFunction) => {\n"
                    "  const statusCode = err.statusCode || 500;\n"
                    "  const message = err.isOperational ? err.message : 'Internal Server Error';\n"
                    "  \n"
                    "  logger.error({\n"
                    "    error: err.message,\n"
                    "    stack: err.stack,\n"
                    "    url: req.url,\n"
                    "    method: req.method\n"
                    "  });\n"
                    "  \n"
                    "  res.status(statusCode).json({ error: message });\n"
                    "};"
                ],
                antipatterns=["exposing stack traces", "no logging", "generic error messages"],
                resonance=0.88,
                source="express_typescript_pack"
            ),

            "typescript_service_layer": CodePattern(
                name="typescript_service_layer",
                summary="Service layer with dependency injection and interfaces",
                apis=["interface", "class", "constructor", "async", "Repository"],
                snippets=[
                    "interface UserRepository {\n"
                    "  findById(id: string): Promise<User | null>;\n"
                    "  create(user: CreateUserDto): Promise<User>;\n"
                    "  update(id: string, data: UpdateUserDto): Promise<User>;\n"
                    "}\n\n"
                    "class UserService {\n"
                    "  constructor(\n"
                    "    private userRepository: UserRepository,\n"
                    "    private logger: Logger\n"
                    "  ) {}\n"
                    "  \n"
                    "  async createUser(userData: CreateUserDto): Promise<User> {\n"
                    "    try {\n"
                    "      this.logger.info('Creating user', { email: userData.email });\n"
                    "      return await this.userRepository.create(userData);\n"
                    "    } catch (error) {\n"
                    "      this.logger.error('Failed to create user', error);\n"
                    "      throw new AppError('User creation failed', 400);\n"
                    "    }\n"
                    "  }\n"
                    "}"
                ],
                antipatterns=["tight coupling", "no error handling", "missing interfaces"],
                resonance=0.91,
                source="express_typescript_pack"
            )
        }

    @staticmethod
    def get_spring_java_pack() -> Dict[str, CodePattern]:
        """Spring Boot + Java patterns"""
        return {
            "spring_rest_controller": CodePattern(
                name="spring_rest_controller",
                summary="RESTful controller with validation and exception handling",
                apis=["@RestController", "@RequestMapping", "@PostMapping", "@Valid", "ResponseEntity"],
                snippets=[
                    "@RestController\n"
                    "@RequestMapping(\"/api/users\")\n"
                    "@Validated\n"
                    "public class UserController {\n"
                    "    \n"
                    "    @Autowired\n"
                    "    private UserService userService;\n"
                    "    \n"
                    "    @PostMapping\n"
                    "    public ResponseEntity<UserDto> createUser(\n"
                    "            @Valid @RequestBody CreateUserRequest request) {\n"
                    "        try {\n"
                    "            UserDto user = userService.createUser(request);\n"
                    "            return ResponseEntity.status(HttpStatus.CREATED).body(user);\n"
                    "        } catch (ValidationException e) {\n"
                    "            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage());\n"
                    "        }\n"
                    "    }\n"
                    "}"
                ],
                antipatterns=["missing validation", "no exception handling", "direct entity exposure"],
                resonance=0.93,
                source="spring_java_pack"
            ),

            "spring_service_layer": CodePattern(
                name="spring_service_layer",
                summary="Service layer with transaction management and logging",
                apis=["@Service", "@Transactional", "@Autowired", "Logger", "Repository"],
                snippets=[
                    "@Service\n"
                    "@Transactional\n"
                    "public class UserService {\n"
                    "    \n"
                    "    private static final Logger logger = LoggerFactory.getLogger(UserService.class);\n"
                    "    \n"
                    "    @Autowired\n"
                    "    private UserRepository userRepository;\n"
                    "    \n"
                    "    @Autowired\n"
                    "    private PasswordEncoder passwordEncoder;\n"
                    "    \n"
                    "    public UserDto createUser(CreateUserRequest request) {\n"
                    "        logger.info(\"Creating user with email: {}\", request.getEmail());\n"
                    "        \n"
                    "        if (userRepository.existsByEmail(request.getEmail())) {\n"
                    "            throw new UserAlreadyExistsException(\"User with email already exists\");\n"
                    "        }\n"
                    "        \n"
                    "        User user = new User();\n"
                    "        user.setEmail(request.getEmail());\n"
                    "        user.setPassword(passwordEncoder.encode(request.getPassword()));\n"
                    "        \n"
                    "        User savedUser = userRepository.save(user);\n"
                    "        logger.info(\"User created successfully with ID: {}\", savedUser.getId());\n"
                    "        \n"
                    "        return UserDto.fromEntity(savedUser);\n"
                    "    }\n"
                    "}"
                ],
                antipatterns=["missing @Transactional", "no logging", "password in plain text"],
                resonance=0.90,
                source="spring_java_pack"
            ),

            "spring_jpa_repository": CodePattern(
                name="spring_jpa_repository",
                summary="JPA repository with custom queries and specifications",
                apis=["@Repository", "JpaRepository", "@Query", "Pageable", "Specification"],
                snippets=[
                    "@Repository\n"
                    "public interface UserRepository extends JpaRepository<User, Long>, JpaSpecificationExecutor<User> {\n"
                    "    \n"
                    "    Optional<User> findByEmail(String email);\n"
                    "    \n"
                    "    boolean existsByEmail(String email);\n"
                    "    \n"
                    "    @Query(\"SELECT u FROM User u WHERE u.createdAt >= :since AND u.active = true\")\n"
                    "    Page<User> findActiveUsersSince(@Param(\"since\") LocalDateTime since, Pageable pageable);\n"
                    "    \n"
                    "    @Modifying\n"
                    "    @Query(\"UPDATE User u SET u.lastLoginAt = :loginTime WHERE u.id = :userId\")\n"
                    "    void updateLastLogin(@Param(\"userId\") Long userId, @Param(\"loginTime\") LocalDateTime loginTime);\n"
                    "}"
                ],
                antipatterns=["missing @Repository", "no pagination", "N+1 query problems"],
                resonance=0.87,
                source="spring_java_pack"
            ),

            "spring_security_config": CodePattern(
                name="spring_security_config",
                summary="Security configuration with JWT and method security",
                apis=["@Configuration", "@EnableWebSecurity", "SecurityFilterChain", "JwtAuthenticationFilter"],
                snippets=[
                    "@Configuration\n"
                    "@EnableWebSecurity\n"
                    "@EnableMethodSecurity\n"
                    "public class SecurityConfig {\n"
                    "    \n"
                    "    @Autowired\n"
                    "    private JwtAuthenticationEntryPoint jwtAuthenticationEntryPoint;\n"
                    "    \n"
                    "    @Autowired\n"
                    "    private JwtAuthenticationFilter jwtAuthenticationFilter;\n"
                    "    \n"
                    "    @Bean\n"
                    "    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {\n"
                    "        http.csrf(csrf -> csrf.disable())\n"
                    "            .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))\n"
                    "            .authorizeHttpRequests(auth -> auth\n"
                    "                .requestMatchers(\"/api/auth/**\").permitAll()\n"
                    "                .requestMatchers(\"/api/public/**\").permitAll()\n"
                    "                .anyRequest().authenticated()\n"
                    "            )\n"
                    "            .exceptionHandling(ex -> ex.authenticationEntryPoint(jwtAuthenticationEntryPoint))\n"
                    "            .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);\n"
                    "        \n"
                    "        return http.build();\n"
                    "    }\n"
                    "}"
                ],
                antipatterns=["permitting all requests", "missing CSRF protection", "weak authentication"],
                resonance=0.85,
                source="spring_java_pack"
            )
        }

    @staticmethod
    def get_react_advanced_pack() -> Dict[str, CodePattern]:
        """Advanced React + TypeScript patterns"""
        return {
            "react_custom_hook": CodePattern(
                name="react_custom_hook",
                summary="Reusable custom hook with TypeScript generics and error handling",
                apis=["useState", "useEffect", "useCallback", "useMemo", "generic"],
                snippets=[
                    "interface UseApiOptions {\n"
                    "  immediate?: boolean;\n"
                    "  onSuccess?: (data: any) => void;\n"
                    "  onError?: (error: Error) => void;\n"
                    "}\n"
                    "\n"
                    "interface UseApiResult<T> {\n"
                    "  data: T | null;\n"
                    "  loading: boolean;\n"
                    "  error: Error | null;\n"
                    "  execute: () => Promise<void>;\n"
                    "  reset: () => void;\n"
                    "}\n"
                    "\n"
                    "function useApi<T>(\n"
                    "  url: string,\n"
                    "  options: UseApiOptions = {}\n"
                    "): UseApiResult<T> {\n"
                    "  const [data, setData] = useState<T | null>(null);\n"
                    "  const [loading, setLoading] = useState(false);\n"
                    "  const [error, setError] = useState<Error | null>(null);\n"
                    "  \n"
                    "  const execute = useCallback(async () => {\n"
                    "    try {\n"
                    "      setLoading(true);\n"
                    "      setError(null);\n"
                    "      \n"
                    "      const response = await fetch(url);\n"
                    "      if (!response.ok) {\n"
                    "        throw new Error(`HTTP error! status: ${response.status}`);\n"
                    "      }\n"
                    "      \n"
                    "      const result = await response.json();\n"
                    "      setData(result);\n"
                    "      options.onSuccess?.(result);\n"
                    "    } catch (err) {\n"
                    "      const error = err instanceof Error ? err : new Error('Unknown error');\n"
                    "      setError(error);\n"
                    "      options.onError?.(error);\n"
                    "    } finally {\n"
                    "      setLoading(false);\n"
                    "    }\n"
                    "  }, [url, options]);\n"
                    "  \n"
                    "  const reset = useCallback(() => {\n"
                    "    setData(null);\n"
                    "    setError(null);\n"
                    "    setLoading(false);\n"
                    "  }, []);\n"
                    "  \n"
                    "  useEffect(() => {\n"
                    "    if (options.immediate !== false) {\n"
                    "      execute();\n"
                    "    }\n"
                    "  }, [execute, options.immediate]);\n"
                    "  \n"
                    "  return { data, loading, error, execute, reset };\n"
                    "}"
                ],
                antipatterns=["missing error handling", "no loading states", "untyped responses"],
                resonance=0.94,
                source="react_advanced_pack"
            ),

            "react_context_provider": CodePattern(
                name="react_context_provider",
                summary="Type-safe context provider with reducer pattern",
                apis=["createContext", "useContext", "useReducer", "Provider", "Context"],
                snippets=[
                    "interface User {\n"
                    "  id: string;\n"
                    "  email: string;\n"
                    "  name: string;\n"
                    "}\n"
                    "\n"
                    "interface AuthState {\n"
                    "  user: User | null;\n"
                    "  isAuthenticated: boolean;\n"
                    "  loading: boolean;\n"
                    "}\n"
                    "\n"
                    "type AuthAction =\n"
                    "  | { type: 'AUTH_START' }\n"
                    "  | { type: 'AUTH_SUCCESS'; payload: User }\n"
                    "  | { type: 'AUTH_FAILURE' }\n"
                    "  | { type: 'LOGOUT' };\n"
                    "\n"
                    "interface AuthContextType extends AuthState {\n"
                    "  login: (email: string, password: string) => Promise<void>;\n"
                    "  logout: () => void;\n"
                    "}\n"
                    "\n"
                    "const AuthContext = createContext<AuthContextType | undefined>(undefined);\n"
                    "\n"
                    "function authReducer(state: AuthState, action: AuthAction): AuthState {\n"
                    "  switch (action.type) {\n"
                    "    case 'AUTH_START':\n"
                    "      return { ...state, loading: true };\n"
                    "    case 'AUTH_SUCCESS':\n"
                    "      return {\n"
                    "        user: action.payload,\n"
                    "        isAuthenticated: true,\n"
                    "        loading: false\n"
                    "      };\n"
                    "    case 'AUTH_FAILURE':\n"
                    "      return {\n"
                    "        user: null,\n"
                    "        isAuthenticated: false,\n"
                    "        loading: false\n"
                    "      };\n"
                    "    case 'LOGOUT':\n"
                    "      return {\n"
                    "        user: null,\n"
                    "        isAuthenticated: false,\n"
                    "        loading: false\n"
                    "      };\n"
                    "    default:\n"
                    "      return state;\n"
                    "  }\n"
                    "}\n"
                    "\n"
                    "export function AuthProvider({ children }: { children: React.ReactNode }) {\n"
                    "  const [state, dispatch] = useReducer(authReducer, {\n"
                    "    user: null,\n"
                    "    isAuthenticated: false,\n"
                    "    loading: false\n"
                    "  });\n"
                    "  \n"
                    "  const login = useCallback(async (email: string, password: string) => {\n"
                    "    dispatch({ type: 'AUTH_START' });\n"
                    "    try {\n"
                    "      const user = await authService.login(email, password);\n"
                    "      dispatch({ type: 'AUTH_SUCCESS', payload: user });\n"
                    "    } catch (error) {\n"
                    "      dispatch({ type: 'AUTH_FAILURE' });\n"
                    "      throw error;\n"
                    "    }\n"
                    "  }, []);\n"
                    "  \n"
                    "  const logout = useCallback(() => {\n"
                    "    authService.logout();\n"
                    "    dispatch({ type: 'LOGOUT' });\n"
                    "  }, []);\n"
                    "  \n"
                    "  return (\n"
                    "    <AuthContext.Provider value={{ ...state, login, logout }}>\n"
                    "      {children}\n"
                    "    </AuthContext.Provider>\n"
                    "  );\n"
                    "}\n"
                    "\n"
                    "export function useAuth() {\n"
                    "  const context = useContext(AuthContext);\n"
                    "  if (context === undefined) {\n"
                    "    throw new Error('useAuth must be used within an AuthProvider');\n"
                    "  }\n"
                    "  return context;\n"
                    "}"
                ],
                antipatterns=["missing context validation", "no type safety", "direct state mutation"],
                resonance=0.92,
                source="react_advanced_pack"
            )
        }

    @staticmethod
    def get_all_language_packs() -> Dict[str, CodePattern]:
        """Get all language packs combined"""
        all_patterns = {}

        all_patterns.update(LanguagePacks.get_express_typescript_pack())
        all_patterns.update(LanguagePacks.get_spring_java_pack())
        all_patterns.update(LanguagePacks.get_react_advanced_pack())

        return all_patterns

def main():
    """Demo language packs"""

    print("🌐 Language Packs Demo")
    print("=" * 50)

    # Get all packs
    express_pack = LanguagePacks.get_express_typescript_pack()
    spring_pack = LanguagePacks.get_spring_java_pack()
    react_pack = LanguagePacks.get_react_advanced_pack()

    print(f"\n📦 Express/TypeScript Pack: {len(express_pack)} patterns")
    for name, pattern in express_pack.items():
        print(f"   • {name}: {pattern.summary[:50]}... (r={pattern.resonance:.2f})")

    print(f"\n☕ Spring/Java Pack: {len(spring_pack)} patterns")
    for name, pattern in spring_pack.items():
        print(f"   • {name}: {pattern.summary[:50]}... (r={pattern.resonance:.2f})")

    print(f"\n⚛️  React Advanced Pack: {len(react_pack)} patterns")
    for name, pattern in react_pack.items():
        print(f"   • {name}: {pattern.summary[:50]}... (r={pattern.resonance:.2f})")

    # Show total coverage
    all_patterns = LanguagePacks.get_all_language_packs()
    print(f"\n🎯 Total patterns: {len(all_patterns)}")

    # Show API coverage
    all_apis = set()
    for pattern in all_patterns.values():
        all_apis.update(pattern.apis)

    print(f"🔧 API coverage: {len(all_apis)} unique APIs")
    print(f"   Top APIs: {', '.join(list(all_apis)[:10])}")

if __name__ == "__main__":
    main()